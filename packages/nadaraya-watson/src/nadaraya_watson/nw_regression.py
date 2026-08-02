"""
Nieparametryczna regresja jadrowa Nadarayi-Watsona.

Odpowiednik skryptu MATLAB-owego: funkcja wzorcowa m(x) = atan(2*pi*x),
wejscia z rozkladu trojkatnego na [-1, 1], zaklocenia gaussowskie lub Cauchy'ego.

Uruchomienie:
    python nw_regression.py
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

Kernel = Callable[[NDArray[np.float64]], NDArray[np.float64]]

# --------------------------------------------------------------------------- #
# Jadra
# --------------------------------------------------------------------------- #
# Wszystkie unormowane (calka = 1), zeby dzialaly zarowno w estymatorze
# gestosci, jak i w NW (tam stala normujaca i tak sie skraca).


def rectangular(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Jadro prostokatne na [-1/2, 1/2]. Symetryczne - w odroznieniu od
    wersji [0, 1] z oryginalu, ktora dawala estymator jednostronny."""
    return np.where(np.abs(u) <= 0.5, 1.0, 0.0)


def epanechnikov(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Optymalne w sensie AMISE. Wyraznie gladsze wyniki niz prostokatne."""
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)


def gaussian(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Nosnik nieograniczony - nigdy nie daje pustego okna, wiec brak NaN-ow."""
    return np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)


# --------------------------------------------------------------------------- #
# Estymatory
# --------------------------------------------------------------------------- #


def _kernel_matrix(
    x_eval: NDArray[np.float64],
    X: NDArray[np.float64],
    h: float,
    K: Kernel,
) -> NDArray[np.float64]:
    """Macierz wag K((x_j - X_i)/h) o ksztalcie (len(x_eval), len(X)).

    Dla duzych N * len(x_eval) trzeba to liczyc w kawalkach - patrz uwaga
    na koncu pliku.
    """
    u = (np.asarray(x_eval, dtype=float)[:, None] - np.asarray(X, dtype=float)[None, :]) / h
    return K(u)


def kernel_density(
    x_eval: NDArray[np.float64],
    X: NDArray[np.float64],
    h: float,
    K: Kernel = rectangular,
) -> NDArray[np.float64]:
    """Estymator gestosci Parzena-Rosenblatta.

    f_hat(x) = 1/(N*h) * sum_i K((x - X_i)/h)

    Odpowiednik `kernel_estimator` z wersji MATLAB-owej. Uwaga: to jest
    estymator gestosci ROZKLADU WEJSC, a nie krok posredni regresji.
    """
    X = np.asarray(X, dtype=float)
    return _kernel_matrix(x_eval, X, h, K).sum(axis=1) / (X.size * h)


def nwkr(
    x_eval: NDArray[np.float64],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    h: float,
    K: Kernel = rectangular,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Estymator Nadarayi-Watsona.

                sum_i K((x - X_i)/h) * Y_i
    m_hat(x) = ----------------------------
                sum_i K((x - X_i)/h)

    Zwraca (m_hat, licznik, mianownik). Tam gdzie okno jest puste,
    m_hat ma NaN - swiadomie, zeby dziura byla widoczna na wykresie
    zamiast udawac zero.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError(f"X i Y musza miec ten sam ksztalt, jest {X.shape} vs {Y.shape}")
    if h <= 0:
        raise ValueError(f"h musi byc dodatnie, jest {h}")

    W = _kernel_matrix(x_eval, X, h, K)
    num = W @ Y
    denom = W.sum(axis=1)

    m_hat = np.divide(num, denom, out=np.full_like(num, np.nan), where=denom != 0)
    return m_hat, num, denom


# --------------------------------------------------------------------------- #
# Regresja lokalnie liniowa (LL) i LOWESS/LOESS
# --------------------------------------------------------------------------- #
# NW dopasowuje lokalnie stala (stopien 0, patrz nwkr). LL dopuszcza nachylenie
# - to usuwa czlon obciazenia zalezny od f'(x)/f(x) i naprawia rzad obciazenia
# na brzegu (O(h^2) zamiast O(h)). LOWESS to LL z adaptacyjnym oknem (span) i
# waga tricube, plus opcjonalne iteracje odpornosciowe Cleveland (1979).


def tricube(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Waga tricube (1-|u|^3)^3 na [-1, 1] - standardowa w LOWESS/LOESS."""
    au = np.abs(u)
    return np.where(au < 1.0, (1.0 - au**3) ** 3, 0.0)


def _bisquare(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Waga bikwadratowa (1-u^2)^2 na (-1, 1) - do przewazania reszt w LOWESS.
    Poza (-1, 1) daje scisle zero, wiec obserwacja odstajaca o > 6*mediana|e|
    wypada z proby przy nastepnej iteracji."""
    return np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)


def _nn_bandwidth(x0: float, X: NDArray[np.float64], q: int) -> float:
    """Odleglosc do q-tego najblizszego sasiada x0 w X - adaptacyjne h LOWESS-u,
    ktore rozszerza sie tam, gdzie dane sa rzadsze."""
    d = np.abs(X - x0)
    q = min(q, X.size)
    return np.partition(d, q - 1)[q - 1]


def _weighted_poly_fit(
    x0: float,
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    weights: NDArray[np.float64],
    degree: int,
) -> float:
    """Wazona regresja wielomianowa stopnia `degree` w otoczeniu x0 (WLS),
    zwraca tylko wyraz wolny, czyli m_hat(x0)."""
    mask = weights > 0
    if mask.sum() < degree + 1:
        return np.nan
    Z = np.vander(X[mask] - x0, degree + 1, increasing=True)
    Wd = weights[mask]
    ZtW = Z.T * Wd
    try:
        beta = np.linalg.solve(ZtW @ Z, ZtW @ Y[mask])
    except np.linalg.LinAlgError:
        return np.nan
    return beta[0]


def local_linear(
    x_eval: NDArray[np.float64],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    h: float,
    K: Kernel = rectangular,
) -> NDArray[np.float64]:
    """Regresja lokalnie liniowa ze stalym h - rdzen LOESS-a, NW to jej
    przypadek szczegolny ze stopniem zero.

    (beta0_hat, beta1_hat) = argmin sum_i K((x-X_i)/h) (Y_i - beta0 - beta1(X_i-x))^2
    m_hat(x) = beta0_hat

    W przeciwienstwie do nwkr dopasowuje w kazdym punkcie wazona prosta, a nie
    srednia - dzieki temu ekstrapoluje poprawnie przy asymetrycznym oknie
    (np. na brzegu dziedziny albo gdy K ma nosnik [0, 1] zamiast [-1, 1]).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    if X.shape != Y.shape:
        raise ValueError(f"X i Y musza miec ten sam ksztalt, jest {X.shape} vs {Y.shape}")
    if h <= 0:
        raise ValueError(f"h musi byc dodatnie, jest {h}")

    m_hat = np.full(x_eval.shape, np.nan)
    for j, x0 in enumerate(x_eval):
        w = K((x0 - X) / h)
        m_hat[j] = _weighted_poly_fit(x0, X, Y, w, degree=1)
    return m_hat


def _robust_fit(
    x_eval: NDArray[np.float64],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    weight_fn: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
    degree: int,
    n_iter: int,
) -> NDArray[np.float64]:
    """Ogolny szkielet iteracji odpornosciowych Cleveland (1979).

    Kazda iteracja: dopasuj w kazdym punkcie danych X_i (waga = weight_fn *
    biezaca waga odpornosciowa), policz reszty e_i, s = mediana|e_i|, nowa
    waga obserwacji = bisquare(e_i / (6s)). Obserwacja odstajaca o wiecej niz
    6 median dostaje wage 0 i znika z kolejnych dopasowan. n_iter=0 pomija
    petle i zwraca zwykle (nieodporne) dopasowanie.

    Wagi koncowej rundy sa przypisane do obserwacji (X_i, Y_i), nie do
    punktu ewaluacji - dopasowanie w x_eval w ostatnim kroku ponownie wazy
    kazda obserwacje jej wlasna waga odpornosciowa. weight_fn(x0, X) moze
    byc oknem o stalym h (robust NW/LL) albo adaptacyjnym span (LOWESS) -
    to odseparowuje efekt samej odpornosci od efektu doboru okna.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    n = X.size

    robust_w = np.ones(n)
    fitted = np.full(n, np.nan)
    for _ in range(n_iter + 1):
        for i in range(n):
            w = weight_fn(X[i], X) * robust_w
            fitted[i] = _weighted_poly_fit(X[i], X, Y, w, degree)
        resid = Y - fitted
        s = np.nanmedian(np.abs(resid))
        # punkty, gdzie lokalne dopasowanie sie nie udalo (za malo obserwacji
        # w oknie) nie maja zdefiniowanej reszty - zerowa waga, a nie NaN,
        # zeby jeden zdegenerowany punkt nie wywalil calej kolejnej iteracji
        if s == 0 or np.isnan(s):
            robust_w = np.where(np.isnan(resid), 0.0, 1.0)
        else:
            robust_w = np.nan_to_num(_bisquare(resid / (6.0 * s)), nan=0.0)

    m_hat = np.full(x_eval.shape, np.nan)
    for j, x0 in enumerate(x_eval):
        w = weight_fn(x0, X) * robust_w
        m_hat[j] = _weighted_poly_fit(x0, X, Y, w, degree)
    return m_hat


def robust_local_poly(
    x_eval: NDArray[np.float64],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    h: float,
    K: Kernel = epanechnikov,
    degree: int = 1,
    n_iter: int = 3,
) -> NDArray[np.float64]:
    """NW (degree=0) albo LL (degree=1) ze STALYM h plus iteracje
    odpornosciowe. W odroznieniu od lowess() ponizej nie zmienia okna na
    adaptacyjne - dzieki temu mierzy dokladnie to, co daje sama odpornosc,
    przy tym samym h co w nwkr/local_linear."""
    def weight_fn(x0: float, Xa: NDArray[np.float64]) -> NDArray[np.float64]:
        return K((x0 - Xa) / h)

    return _robust_fit(x_eval, X, Y, weight_fn, degree, n_iter)


def lowess(
    x_eval: NDArray[np.float64],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    frac: float = 0.3,
    degree: int = 1,
    n_iter: int = 3,
) -> NDArray[np.float64]:
    """LOWESS/LOESS (Cleveland, 1979): lokalny wielomian stopnia `degree` w
    adaptacyjnym oknie (span = ulamek `frac` najblizszych sasiadow) z waga
    tricube, plus `n_iter` iteracji odpornosciowych (patrz _robust_fit)."""
    X = np.asarray(X, dtype=float)
    n = X.size
    q = max(degree + 1, int(np.ceil(frac * n)))

    def weight_fn(x0: float, Xa: NDArray[np.float64]) -> NDArray[np.float64]:
        h = _nn_bandwidth(x0, Xa, q) or np.finfo(float).eps
        return tricube((Xa - x0) / h)

    return _robust_fit(x_eval, X, Y, weight_fn, degree, n_iter)


# --------------------------------------------------------------------------- #
# Generowanie danych
# --------------------------------------------------------------------------- #


def make_data(
    n: int = 1024,
    noise: str = "gauss",
    scale: float = 0.05,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """X ~ Triangular(-1, 0, 1), Y = atan(2*pi*X) + Z."""
    rng = np.random.default_rng(seed)
    X = rng.triangular(left=-1.0, mode=0.0, right=1.0, size=n)
    X.sort()

    if noise == "gauss":
        Z = scale * rng.standard_normal(n)
    elif noise == "cauchy":
        # odpowiednik trnd(1, ...) w MATLAB - t-Studenta o 1 st. swobody
        Z = scale * rng.standard_cauchy(n)
    else:
        raise ValueError(f"noise musi byc 'gauss' albo 'cauchy', jest {noise!r}")

    return X, m_true(X) + Z


def m_true(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Funkcja wzorcowa."""
    return np.arctan(2.0 * np.pi * np.asarray(x, dtype=float))


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #


def main() -> None:
    n = 1024
    h = 1 / 2**5
    frac = 0.2  # span LOWESS-u: ulamek najblizszych sasiadow
    x = np.arange(-1.0, 1.0 + 1e-9, 0.01)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for row, noise in enumerate(("gauss", "cauchy")):
        X, Y = make_data(n=n, noise=noise, seed=0)

        # --- lewa kolumna: dopasowanie ---
        ax = axes[row, 0]
        ax.scatter(X, Y, s=3, alpha=0.15, color="gray", label="dane")
        ax.plot(x, m_true(x), lw=2, label=r"$m(x)=\arctan(2\pi x)$")

        for K, name in ((rectangular, "prostokatne"), (epanechnikov, "Epanecznikow")):
            m_hat, _, _ = nwkr(x, X, Y, h, K)
            ax.plot(x, m_hat, lw=1.5, label=f"NW, {name}")

        ll_hat = local_linear(x, X, Y, h, epanechnikov)
        ax.plot(x, ll_hat, lw=1.5, ls="--", label="LL, Epanecznikow")

        lowess_hat = lowess(x, X, Y, frac=frac, degree=1, n_iter=3)
        ax.plot(x, lowess_hat, lw=1.5, ls="-.", label="LOWESS (odporny)")

        ax.set_title(f"zaklocenia: {noise}, h = 1/{int(1/h)}")
        ax.set_ylim(-3, 3)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # --- prawa kolumna: gestosc / mianownik ---
        ax = axes[row, 1]
        f_hat = kernel_density(x, X, h, rectangular)
        _, _, denom = nwkr(x, X, Y, h, rectangular)
        ax.plot(x, f_hat, label=r"$\hat{f}(x)$ (Parzen-Rosenblatt)")
        ax.plot(x, denom / (n * h), "--", label=r"mianownik NW $/(Nh)$")
        ax.plot(x, np.where(np.abs(x) <= 1, 1 - np.abs(x), 0), ":", label="prawdziwa gestosc")
        ax.set_title("kontrola: mianownik NW = $N h \\hat{f}(x)$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[1]:
        ax.set_xlabel("x")

    fig.tight_layout()
    fig.savefig("nw_regression.png", dpi=130)
    print("zapisano nw_regression.png")

    # blad sredniokwadratowy na siatce: rozbite na DWA niezalezne efekty -
    # (a) stopien wielomianu / szerokosc okna, (b) same iteracje odpornosciowe
    # przy TYM SAMYM h co NW, zeby nie mieszac obu efektow w jedna liczbe.
    print()
    for noise in ("gauss", "cauchy"):
        X, Y = make_data(n=n, noise=noise, seed=0)
        truth = m_true(x)

        def mse_of(m_hat: NDArray[np.float64]) -> float:
            return float(np.nanmean((m_hat - truth) ** 2))

        rows: list[tuple[str, float]] = []
        for K, name in ((rectangular, "NW prostokatne"), (epanechnikov, "NW Epanecznikow")):
            m_hat, _, _ = nwkr(x, X, Y, h, K)
            rows.append((name, mse_of(m_hat)))

        rows.append(("LL Epanecznikow", mse_of(local_linear(x, X, Y, h, epanechnikov))))
        rows.append((
            "NW + odpornosc (h staly)",
            mse_of(robust_local_poly(x, X, Y, h, epanechnikov, degree=0, n_iter=3)),
        ))
        rows.append((
            "LL + odpornosc (h staly)",
            mse_of(robust_local_poly(x, X, Y, h, epanechnikov, degree=1, n_iter=3)),
        ))
        rows.append(("LOWESS, span (it=0)", mse_of(lowess(x, X, Y, frac=frac, degree=1, n_iter=0))))
        rows.append(("LOWESS, span (it=3)", mse_of(lowess(x, X, Y, frac=frac, degree=1, n_iter=3))))

        for name, mse in rows:
            print(f"{noise:7s} / {name:26s}: MSE = {mse:.4f}")
        print()

    # --- kontrola: statsmodels na tych samych danych ---
    # KernelReg(reg_type='lc') to doslownie nwkr (local constant = NW),
    # reg_type='ll' to local_linear. statsmodels.lowess ma wbudowane
    # iteracje odpornosciowe (parametr `it`).
    print("--- kontrola: statsmodels (te same dane, seed=0) ---")
    from statsmodels.nonparametric.kernel_regression import KernelReg
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

    for noise in ("gauss", "cauchy"):
        X, Y = make_data(n=n, noise=noise, seed=0)
        truth = m_true(x)

        nw_sm = KernelReg(endog=Y, exog=X, var_type="c", reg_type="lc", bw=[h])
        m_nw_sm, _ = nw_sm.fit(x)
        mse_nw_sm = float(np.nanmean((m_nw_sm - truth) ** 2))

        ll_sm = KernelReg(endog=Y, exog=X, var_type="c", reg_type="ll", bw=[h])
        m_ll_sm, _ = ll_sm.fit(x)
        mse_ll_sm = float(np.nanmean((m_ll_sm - truth) ** 2))

        fit = sm_lowess(Y, X, frac=frac, it=3, xvals=x)
        mse_lowess_sm = float(np.nanmean((fit - truth) ** 2))

        print(f"{noise:7s} / {'statsmodels NW (lc)':26s}: MSE = {mse_nw_sm:.4f}")
        print(f"{noise:7s} / {'statsmodels LL (ll)':26s}: MSE = {mse_ll_sm:.4f}")
        print(f"{noise:7s} / {'statsmodels lowess it=3':26s}: MSE = {mse_lowess_sm:.4f}")
        print()


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------- #
# Uwaga o pamieci
# --------------------------------------------------------------------------- #
# _kernel_matrix alokuje tablice (len(x_eval), N) w float64. Przy 201 x 1024
# to 1.6 MB - bez znaczenia. Przy N = 10^6 i tysiacu punktow ewaluacji byloby
# 8 GB, wiec wtedy trzeba petli po blokach x_eval albo - dla jader o zwartym
# nosniku - scipy.spatial.cKDTree / np.searchsorted na posortowanym X, zeby
# wybierac tylko sasiadow w promieniu h.
