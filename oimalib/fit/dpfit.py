"""
@author: Anthony Soulain (University of Sydney)

-------------------------------------------------------------------------
OPTIMAL: OPTical Interferometry Modelisation and Analysis Library
-------------------------------------------------------------------------

Fitting tools (developped by A. Merand).

--------------------------------------------------------------------
"""

import contextlib
import time
from functools import reduce

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt

verboseTime = time.time()


def leastsqFit(
    func,
    x,
    params,
    y,
    err=None,
    fitOnly=None,
    verbose=False,
    doNotFit=None,
    epsfcn=1e-7,
    ftol=1e-5,
    fullOutput=True,
    normalizedUncer=True,
    follow=None,
    maxfev=5000,
    bounds=None,
):
    """
    - params is a Dict containing the first guess.

    - fits 'y +- err = func(x,params)'. errors are optionnal. in case err is a
      ndarray of 2 dimensions, it is treated as the covariance of the
      errors.

      np.array([[err1**2, 0, .., 0],
                [0, err2**2, 0, .., 0],
                [0, .., 0, errN**2]]) is the equivalent of 1D errors

    - follow=[...] list of parameters to "follow" in the fit, i.e. to print in
      verbose mode

        if not (np.isscalar(params[k]) and not isinstance(params[k], str)):
      parameters in 'params'. Alternatively, one can give a list of
      parameters not to be fitted, as 'doNotFit='

    - doNotFit has a similar purpose: for example if params={'a0':,
      'a1': 'b1':, 'b2':}, doNotFit=['a'] will result in fitting only
      'b1' and 'b2'. WARNING: if you name parameter 'A' and another one 'AA',
      you cannot use doNotFit to exclude only 'A' since 'AA' will be excluded as
      well...

    - normalizedUncer=True: the uncertainties are independent of the Chi2, in
      other words the uncertainties are scaled to the Chi2. If set to False, it
      will trust the values of the error bars: it means that if you grossely
      underestimate the data's error bars, the uncertainties of the parameters
      will also be underestimated (and vice versa).

    - bounds = dictionnary with lower/upper bounds. if bounds are not specified,
        (-inf/inf will be used)

    - verbose:
        True (or 1): show progress
        2: show progress and best fit with errors
        3: show progress, best fit with errors and correlations

    returns dictionary with:
    'best': bestparam,
    'uncer': uncertainties,
    'chi2': chi2_reduced,
    'model': func(x, bestparam)
    'cov': covariance matrix (normalized if normalizedUncer)
    'fitOnly': names of the columns of 'cov'
    """
    global Ncalls, pfitKeys, pfix, _func, data_errors, trackP

    if doNotFit is None:
        doNotFit = []
    if bounds is None:
        bounds = {}
    # -- fit all parameters by default
    if fitOnly is None:
        if len(doNotFit) > 0:
            fitOnly = filter(lambda x: x not in doNotFit, params.keys())
        else:
            fitOnly = params.keys()
        fitOnly = list(fitOnly)
        fitOnly.sort()  # makes some display nicer

    # -- check that all parameters are numbers
    NaNs = []
    for k in fitOnly:
        # if not (type(params[k])==float or type(params[k])==int):
        if not (np.isscalar(params[k]) and not isinstance(params[k], str)):
            NaNs.append(k)
    fitOnly = sorted(list(filter(lambda x: x not in NaNs, fitOnly)))

    # -- build fitted parameters vector:
    pfit = [params[k] for k in fitOnly]

    # -- built fixed parameters dict:
    pfix = {}
    for k in params:
        if k not in fitOnly:
            pfix[k] = params[k]
    if verbose:
        print(f"[dpfit] {len(fitOnly)} FITTED parameters:", end=" ")
        if len(fitOnly) < 100 or (isinstance(verbose, int) and verbose > 1):
            print(fitOnly)
            print("[dpfit] epsfcn=", epsfcn, "ftol=", ftol)
        else:
            print(" ")

    # -- actual fit
    Ncalls = 0
    trackP = {}
    t0 = time.time()
    mesg = ""
    if np.iterable(err) and len(np.array(err).shape) == 2:
        # -- assumes err matrix is co-covariance
        _func = func
        pfitKeys = fitOnly
        plsq, cov = scipy.optimize.curve_fit(
            _fitFunc2, x, y, pfit, sigma=err, epsfcn=epsfcn, ftol=ftol
        )
        info, mesg, _ier = (
            {"nfev": Ncalls, "exec time": time.time() - t0},
            "curve_fit",
            None,
        )
    elif bounds is None or bounds == {}:
        # ==== LEGACY! ===========================
        if verbose:
            print("[dpfit] using scipy.optimize.leastsq")
        plsq, cov, info, mesg, _ier = scipy.optimize.leastsq(
            _fitFunc,
            pfit,
            args=(
                fitOnly,
                x,
                y,
                err,
                func,
                pfix,
                verbose,
                follow,
            ),
            full_output=True,
            epsfcn=epsfcn,
            ftol=ftol,
            maxfev=maxfev,
        )
        info["exec time"] = time.time() - t0
        mesg = mesg.replace("\n", "")
    else:
        method = "L-BFGS-B"

        # method = 'SLSQP'
        # method = 'TNC'
        # method = 'trust-constr'
        if verbose:
            print(f"[dpfit] using scipy.optimize.minimize ({method})")
        Bounds = []
        for k in fitOnly:
            if k in bounds:
                Bounds.append(bounds[k])
            else:
                Bounds.append((-np.inf, np.inf))

        result = scipy.optimize.minimize(
            _fitFuncMin,
            pfit,
            tol=ftol,
            options={"maxiter": maxfev},
            bounds=Bounds,
            method=method,
            args=(
                fitOnly,
                x,
                y,
                err,
                func,
                pfix,
                verbose,
                follow,
            ),
        )
        plsq = result.x
        # display(result)
        try:
            # https://github.com/scipy/scipy/blob/2526df72e5d4ca8bad6e2f4b3cbdfbc33e805865/scipy/optimize/minpack.py#L739
            # Do Moore-Penrose inverse discarding zero singular values.
            _, s, VT = np.linalg.svd(result.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
            if verbose:
                print("[dpfit] zeros in cov?", any(s <= threshold))
            s = s[s > threshold]
            VT = VT[: s.size]
            cov = np.dot(VT.T / s**2, VT)
        except Exception:
            cov = np.zeros((len(fitOnly), len(fitOnly)))
        # ------------------------------------------------------
        info = {"nfev": Ncalls, "exec time": time.time() - t0}
        mesg, _ = result.message, None

    if verbose:
        print("[dpfit]", mesg)
        # print('[dpfit] ier:', ier)
        print("[dpfit]", info["nfev"], "function calls", end=" ")
        t = 1000 * info["exec time"] / info["nfev"]
        n = -int(np.log10(t)) + 3
        print("(", round(t, n), "ms on average)")

    notsig = []
    if cov is None:
        if verbose:
            print("[dpfit] \033[31mWARNING: singular covariance matrix,", end=" ")
            print("uncertainties cannot be computed\033[0m")
        mesg += "; singular covariance matrix"
        # print('       ', info['fjac'].shape)
        # -- try to figure out what is going on!
        delta = np.array(pfit) - np.array(plsq)
        for i, k in enumerate(fitOnly):
            if "fjac" in info:
                # test = max(np.abs(info['fjac'][i,:]))==0
                _i = list(info["ipvt"]).index(i + 1)
                test = max(np.abs(info["fjac"][_i, :])) == 0
            else:
                test = np.abs(delta[i]) <= epsfcn
            if test:
                if verbose:
                    print(
                        '[dpfit] \033[31m         parameter "' + k + '" does not change CHI2:',
                        end=" ",
                    )
                    print("IT CANNOT BE FITTED\033[0m")
                mesg += '; parameter "' + k + '" does not change CHI2'
                notsig.append(k)
        cov = np.zeros((len(fitOnly), len(fitOnly)))

    # -- best fit -> agregate to pfix
    for i, k in enumerate(fitOnly):
        pfix[k] = plsq[i]

    # -- reduced chi2
    model = func(x, pfix)
    # -- residuals
    if np.iterable(err) and len(np.array(err).shape) == 2:
        # -- assumes err matrix is co-covariance
        r = y - model
        chi2 = np.dot(np.dot(np.transpose(r), np.linalg.inv(err)), r)
        ndof = len(y) - len(pfit) + 1
        reducedChi2 = chi2 / ndof
    else:
        tmp = _fitFunc(plsq, fitOnly, x, y, err, func, pfix)
        try:
            chi2 = (np.array(tmp) ** 2).sum()
        except Exception:
            chi2 = 0.0
            for x in tmp:
                chi2 += np.sum(x**2)

        ndof = np.sum([1 if np.isscalar(i) else len(i) for i in tmp]) - len(pfit) + 1
        reducedChi2 = chi2 / ndof
        if not np.isscalar(reducedChi2):
            reducedChi2 = np.mean(reducedChi2)

    if normalizedUncer:
        with contextlib.suppress(Exception):
            cov *= reducedChi2

    # -- uncertainties:
    uncer = {}
    for k in pfix:
        if k not in fitOnly:
            uncer[k] = 0  # not fitted, uncertatinties to 0
        else:
            i = fitOnly.index(k)
            if cov is None:
                uncer[k] = -1
            else:
                uncer[k] = np.sqrt(np.abs(np.diag(cov)[i]))

    # -- simple criteria to see if step is too large
    notconverg = []
    for k in filter(lambda x: x != "reduced chi2", trackP.keys()):
        n = len(trackP[k])
        std2 = np.std(trackP[k][(3 * n) // 4 :])
        _ = np.ptp(trackP[k][(3 * n) // 4 :])
        if std2 > 2 * uncer[k] and k not in notsig:
            notconverg.append(k)
    if notconverg and verbose:
        print(
            "[dpfit] \033[33mParameters",
            notconverg,
            "may not be converging properly\033[0m",
        )
        print(
            '[dpfit] \033[33mcheck with "showFit" '
            + "(too sensitive to relative variations?)\033[0m"
        )

    if isinstance(verbose, int) and verbose > 1:
        # print('-'*30)
        print("# --     CHI2=", chi2)
        print("# -- red CHI2=", reducedChi2)
        print("# --     NDOF=", int(chi2 / reducedChi2))
        # print('-'*30)
        dispBest({"best": pfix, "uncer": uncer, "fitOnly": fitOnly})

    # -- result:
    if fullOutput:
        diag_cov = np.diag(cov).copy()
        diag_cov[diag_cov < 0] = 0.0
        cor = np.sqrt(diag_cov)
        cor = cor[:, None] * cor[None, :]
        cor[cor == 0] = 1e-6
        cor = cov / cor
        for k in trackP:
            trackP[k] = np.array(trackP[k])

        pfix = {
            "func": func,
            "best": pfix,
            "uncer": uncer,
            "chi2": reducedChi2,
            "model": model,
            "cov": cov,
            "fitOnly": fitOnly,
            "epsfcn": epsfcn,
            "ftol": ftol,
            "info": info,
            "cor": cor,
            "x": x,
            "y": y,
            "ndof": ndof,
            "doNotFit": doNotFit,
            "covd": {
                ki: {kj: cov[i, j] for j, kj in enumerate(fitOnly)} for i, ki in enumerate(fitOnly)
            },
            "cord": {
                ki: {kj: cor[i, j] for j, kj in enumerate(fitOnly)} for i, ki in enumerate(fitOnly)
            },
            "normalized uncertainties": normalizedUncer,
            "maxfev": maxfev,
            "firstGuess": params,
            "track": trackP,
            "mesg": mesg,
            "not significant": notsig,
            "not converging": notconverg,
            "chi2_real": chi2,
        }
        if isinstance(verbose, int) and verbose > 2 and np.size(cor) > 1:
            dispCor(pfix)
    return pfix


def _fitFunc(pfit, pfitKeys, x, y, err=None, func=None, pfix=None, verbose=False, follow=None):
    """
    interface  scipy.optimize.leastsq:
    - x,y,err are the data to fit: f(x) = y +- err
    - pfit is a list of the paramters
    - pfitsKeys are the keys to build the dict
    pfit and pfix (optional) and combines the two
    in 'A', in order to call F(X,A)

    in case err is a ndarray of 2 dimensions, it is treated as the
    covariance of the errors.
    np.array([[err1**2, 0, .., 0],
             [ 0, err2**2, 0, .., 0],
             [0, .., 0, errN**2]]) is the equivalent of 1D errors
    """
    global verboseTime, Ncalls, trackP
    Ncalls += 1

    params = {}
    # -- build dic from parameters to fit and their values:
    for i, k in enumerate(pfitKeys):
        params[k] = pfit[i]
    # -- complete with the non fitted parameters:
    for k in pfix:
        params[k] = pfix[k]
    if err is None:
        err = np.ones(np.array(y).shape)

    # -- compute residuals
    if (
        (type(y) is np.ndarray and type(err) is np.ndarray)
        or (np.isscalar(y) and type(err) is np.ndarray)
        or (type(y) is np.ndarray and np.isscalar(err))
        or (np.isscalar(y) and np.isscalar(err))
    ):
        model = func(x, params)
        res = ((np.array(y) - model) / err).flatten()
    else:
        # much slower: this time assumes y (and the result from func) is
        # a list of things, each convertible in np.array
        res = []
        tmp = func(x, params)
        if np.isscalar(err):
            err = 0 * y + err
        # print 'DEBUG:', tmp.shape, y.shape, err.shape

        for k in range(len(y)):
            df = (np.array(tmp[k]) - np.array(y[k])) / np.array(err[k])
            try:
                res.extend(list(df))
            except Exception:
                res.append(df)

    try:
        chi2 = (res**2).sum / (len(res) - len(pfit) + 1.0)
    except Exception:
        # list of elements
        chi2 = 0
        N = 0
        # res2 = []
        for r in res:
            if np.isscalar(r):
                chi2 += r**2
                N += 1
                # r  # es2.append(r)
            else:
                chi2 += np.sum(np.array(r) ** 2)
                N += len(r)
                # res2.extend(list(r))
        chi2 /= N - len(pfit) + 1.0

    if verbose and time.time() > (verboseTime + 10):
        verboseTime = time.time()
        print(
            "[dpfit]",
            time.asctime(),
            f"{Ncalls:03d}/{int(Ncalls / len(pfit)):03d}",
            end=" ",
        )
        print(f"CHI2: {chi2:6.4e}", end="|")
        if follow is None:
            print("")
        else:
            _follow = list(
                filter(
                    lambda x: x in params and type(params[x]) in [float, np.double],
                    follow,
                )
            )
            print("|".join([k + "=" + f"{params[k]:5.2e}" for k in _follow]))
    for i, k in enumerate(pfitKeys):
        if k not in trackP:
            trackP[k] = [pfit[i]]
        else:
            trackP[k].append(pfit[i])
    if "reduced chi2" not in trackP:
        trackP["reduced chi2"] = [chi2]
    else:
        trackP["reduced chi2"].append(chi2)

    return res


def _fitFuncMin(pfit, pfitKeys, x, y, err=None, func=None, pfix=None, verbose=False, follow=None):
    """
    interface  scipy.optimize.minimize:
    - x,y,err are the data to fit: f(x) = y +- err
    - pfit is a list of the paramters
    - pfitsKeys are the keys to build the dict
    pfit and pfix (optional) and combines the two
    in 'A', in order to call F(X,A)

    in case err is a ndarray of 2 dimensions, it is treated as the
    covariance of the errors.
    np.array([[err1**2, 0, .., 0],
             [ 0, err2**2, 0, .., 0],
             [0, .., 0, errN**2]]) is the equivalent of 1D errors
    """
    global verboseTime, Ncalls, trackP
    Ncalls += 1

    params = {}
    # -- build dic from parameters to fit and their values:
    for i, k in enumerate(pfitKeys):
        params[k] = pfit[i]
    # -- complete with the non fitted parameters:
    for k in pfix:
        params[k] = pfix[k]
    if err is None:
        err = np.ones(np.array(y).shape)

    # -- compute residuals
    if isinstance(y, np.ndarray) and isinstance(err, np.ndarray):
        model = func(x, params)
        res = ((np.array(y) - model) / err).flatten()
    else:
        # much slower: this time assumes y (and the result from func) is
        # a list of things, each convertible in np.array
        res = []
        tmp = func(x, params)
        if np.isscalar(err):
            err = 0 * y + err
        # print 'DEBUG:', tmp.shape, y.shape, err.shape

        for k in range(len(y)):
            df = (np.array(tmp[k]) - np.array(y[k])) / np.array(err[k])
            try:
                res.extend(list(df))
            except Exception:
                res.append(df)

    try:
        chi2 = (res**2).sum / (len(res) - len(pfit) + 1.0)
    except Exception:
        # list of elements
        chi2 = 0
        N = 0
        res2 = []
        for r in res:
            if np.isscalar(r):
                chi2 += r**2
                N += 1
                res2.append(r)
            else:
                chi2 += np.sum(np.array(r) ** 2)
                N += len(r)
                res2.extend(list(r))
        res = res2
        chi2 /= float(N - len(pfit) + 1)

    if verbose and time.time() > (verboseTime + 10):
        verboseTime = time.time()
        print("[dpfit]", time.asctime(), f"{Ncalls:5d}", end="")
        print(f"CHI2: {chi2:6.4e}", end="|")
        if follow is None:
            print("")
        else:
            _follow = list(filter(lambda x: x in params, follow))
            print("|".join([k + "=" + f"{params[k]:5.2e}" for k in _follow]))
    return chi2


def _fitFunc2(x, *pfit, verbose=True, follow=None, errs=None):
    """
    for curve_fit
    """
    global pfitKeys, pfix, _func, Ncalls, verboseTime

    if follow is None:
        follow = []
    Ncalls += 1
    params = {}
    # -- build dic from parameters to fit and their values:
    for i, k in enumerate(pfitKeys):
        params[k] = pfit[i]
    # -- complete with the non fitted parameters:
    for k in pfix:
        params[k] = pfix[k]

    res = _func(x, params)

    if verbose and time.time() > (verboseTime + 10):
        verboseTime = time.time()
        print("[dpfit]", time.asctime(), f"{Ncalls:5d}", end="")
        try:
            chi2 = np.sum(res**2) / (len(res) - len(pfit) + 1.0)
            print(f"CHI2: {chi2:6.4e}", end="")
        except Exception:
            # list of elements
            chi2 = 0
            N = 0
            res2 = []
            for r in res:
                if np.isscalar(r):
                    chi2 += r**2
                    N += 1
                    res2.append(r)
                else:
                    chi2 += np.sum(np.array(r) ** 2)
                    N += len(r)
                    res2.extend(list(r))

            res = res2
            print(f"CHI2: {chi2 / float(N - len(pfit) + 1):6.4e}", end=" ")
        if follow is None:
            print("")
        else:
            _follow = list(filter(lambda x: x in params, follow))
            print(" ".join([k + "=" + f"{params[k]:5.2e}" for k in _follow]))

    return res


def dispBest(fit, pre="", asStr=False, asDict=True, color=True):
    # tmp = sorted(fit['best'].keys())
    # -- fitted param:
    tmp = sorted(fit["fitOnly"])
    # -- unfitted:
    tmp += sorted(list(filter(lambda x: x not in fit["fitOnly"], fit["best"].keys())))

    uncer = fit["uncer"]
    pfix = fit["best"]

    res = ""

    maxLength = np.max(np.array([len(k) for k in tmp]))
    format_ = "'%s':" if asDict else "%s"
    # -- write each parameter and its best fit, as well as error
    # -- writes directly a dictionary
    for ik, k in enumerate(tmp):
        padding = " " * (maxLength - len(k))
        formatS = format_ + padding
        formatS = pre + "{" + formatS if ik == 0 and asDict else pre + formatS
        if uncer[k] > 0:
            ndigit = max(-int(np.log10(uncer[k])) + 2, 0)
            if asDict:
                fmt = "%." + str(ndigit) + "f, # +/- %." + str(ndigit) + "f"
            else:
                fmt = "%." + str(ndigit) + "f +/- %." + str(ndigit) + "f"
            col = ("\x1b[94m", "\x1b[0m") if color else ("", "")
            res += col[0] + formatS % k + fmt % (pfix[k], uncer[k]) + col[1] + "\n"
            # print(formatS%k, fmt%(pfix[k], uncer[k]))
        elif uncer[k] == 0:
            col = ("\x1b[97m", "\x1b[0m") if color else ("", "")
            if isinstance(pfix[k], str):
                # print(formatS%k , "'"+pfix[k]+"'", ',')
                res += col[0] + formatS % k + "'" + pfix[k] + "'," + col[1] + "\n"
            else:
                # print(formatS%k , pfix[k], ',')
                res += col[0] + formatS % k + str(pfix[k]) + "," + col[1] + "\n"
        else:
            # print(formatS%k , pfix[k], end='')
            # res += formatS % k + pfix[k]
            # if asDict:
            #     # print(', # +/-', uncer[k])
            #     res += "# +/- " + str(uncer[k]) + "\n"
            # else:
            #     # print('+/-', uncer[k])
            #     res += "+/- " + str(uncer[k]) + "\n"
            pass
    if asDict:
        # print(pre+'}') # end of the dictionnary
        res += pre + "}\n"
    if asStr:
        return res
    else:
        print(res)
        return


def dispCor(fit, ndigit=2, pre="", asStr=False, html=False, maxlen=140):
    # -- parameters names:
    nmax = np.max([len(x) for x in fit["fitOnly"]])
    # -- compact display if too wide
    if maxlen is not None and nmax + (ndigit + 2) * len(fit["fitOnly"]) + 4 > maxlen:
        ndigit = 1
    if maxlen is not None and nmax + (ndigit + 2) * len(fit["fitOnly"]) + 4 > maxlen:
        ndigit = 0

    fmt = "{:>" + str(nmax) + "}"

    def fcount(i):
        return f"{hex(i)[2:]:>2}"

    def fcount2(i):
        return f"{i:2d}"

    if len(fit["fitOnly"]) > 100:
        # -- hexadecimal
        count = fcount  # count = lambda i: "%2s" % hex(i)[2:]
    else:
        # -- decimal
        count = fcount2  # lambda i: "%2d" % i

    if not asStr:
        print(pre + "Correlations (%) ", end=" ")
        print("\033[45m>=90\033[0m", end=" ")
        print("\033[41m>=80\033[0m", end=" ")
        print("\033[43m>=70\033[0m", end=" ")
        print("\033[46m>=50\033[0m", end=" ")
        print("\033[0m>=20\033[0m", end=" ")
        print("\033[37m<20%\033[0m")
        print(pre + " " * (3 + nmax), end=" ")
        for i in range(len(fit["fitOnly"])):
            # print('%2d'%i+' '*(ndigit-1), end=' ')
            c = "\x1b[47m" if i % 2 and ndigit < 2 else "\x1b[0m"
            print(c + count(i) + "\033[0m" + " " * (ndigit), end="")
        print(pre + "")
    elif html:
        res = '<table style="width:100%" border="1">\n'
        res += "<tr>"
        res += "<th>" + pre + " " * (2 + ndigit + nmax) + " </th>"
        for i in range(len(fit["fitOnly"])):
            res += f"<th>{i:2d}" + " " * (ndigit) + "</th>"
        res += "</tr>\n"
    else:
        res = pre + " " * (3 + ndigit // 2 + nmax) + " "
        for i in range(len(fit["fitOnly"])):
            # res += '%2d'%i+' '*(ndigit)
            res += count(i) + " " * (ndigit)
        res += "\n"

    for i, p in enumerate(fit["fitOnly"]):
        c = "\x1b[47m" if i % 2 and ndigit < 2 else "\x1b[0m"
        if not asStr:
            print(pre + c + count(i) + ":" + fmt.format(p) + "\033[0m", end=" ")
        elif html:
            res += f"<tr>\n<td>{pre}{count(i)}:{fmt.format(p)}</td >\n"
        else:
            res += pre + count(i) + ":" + fmt.format(p) + " "

        for j, x in enumerate(fit["cor"][i, :]):
            c = "\x1b[2m" if i == j else "\x1b[0m"
            hcol = "#FFFFFF"
            if i != j:
                if abs(x) >= 0.9:
                    col = "\033[45m"
                    hcol = "#FF66FF"
                elif abs(x) >= 0.8:
                    col = "\033[41m"
                    hcol = "#FF6666"
                elif abs(x) >= 0.7:
                    col = "\033[43m"
                    hcol = "#FFEE66"
                elif abs(x) >= 0.5:
                    col = "\033[46m"
                    hcol = "#CCCCCC"
                elif abs(x) < 0.2:
                    col = "\033[37m"
                    hcol = "#FFFFFF"
                else:
                    col = ""
            elif i % 2 and ndigit < 2:
                col = "\033[47m"
            else:
                col = "\033[0m"
            # tmp = fmtd%x
            # tmp = tmp.replace('0.', '.')
            # tmp = tmp.replace('1.'+'0'*ndigit, '1.')
            # if i==j:
            #     tmp = '#'*(2+ndigit)
            # print(c+col+tmp+'\033[0m', end=' ')
            if i == j:
                tmp = "#" * (1 + ndigit)
            else:
                if ndigit == 0:
                    # tmp = '%2d'%int(round(10*x, 0))
                    tmp = "-" if x < 0 else "+"
                if ndigit == 1:
                    # tmp = '%2d'%int(round(10*x, 0))
                    tmp = "--" if x < 0 else "++"
                elif ndigit == 2:
                    tmp = f"{int(round(100 * x, 0)):3d}"

            if not asStr:
                print(c + col + tmp + "\033[0m", end=" ")
            elif html:
                res += f'<td bgcolor="{hcol}">' + tmp + "</td>\n"
            else:
                res += tmp + " "
        if not asStr:
            print("")
        elif html:
            res += "</tr>\n"
        else:
            res += "\n"
    if html:
        res += "</table>\n"
    if asStr:
        return res


def factors(n):
    """
    returns the factirs for integer n
    """
    return list(
        set(
            reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
        )
    )


def subp(N, imax=None):
    """
    gives the dimensions of a gird of plots for N plots, allow up to imax empty
    plots to have a squarish grid
    """
    if N == 2:
        return (1, 2)
    if imax is None:
        imax = max(2, N // 5)
    S = {}
    for i in range(imax + 1):
        F = np.array(factors(N + i))
        S[N + i] = sorted(F[np.argsort(np.abs(F - np.sqrt(N)))][:2])
        if N + i == int(np.sqrt(N + i)) ** 2:
            return [int(np.sqrt(N + i)), int(np.sqrt(N + i))]
    # -- get the one with most squarish aspect ratio
    K = list(S.keys())
    R = [S[k][1] / S[k][0] for k in K]
    _ = K[np.argmin(R)]
    return S[K[np.argmin(R)]]


def _callbackAxes(ax):
    """
    make sure y ranges follows the data, after x range has been adjusted
    """
    global AX, T
    xlim = ax.get_xlim()
    x = np.arange(len(T["reduced chi2"]))
    w = (x >= xlim[0]) * (x <= xlim[1])
    for k in AX:
        if k == "reduced chi2" and np.max(T[k][w]) / np.min(T[k][w]) > 100:
            AX[k].set_yscale("log")
            AX[k].set_ylim(0.9 * np.min(T[k][w]), 1.1 * np.max(T[k][w]))
        else:
            AX[k].set_yscale("linear")
            AX[k].set_ylim(
                np.min(T[k][w]) - 0.1 * np.ptp(T[k][w]),
                np.max(T[k][w]) + 0.1 * np.ptp(T[k][w]),
            )


def showFit(fit, fig=99):
    """
    plot the evolution of the fitted parameters as function of iteration,
    as well as chi2
    """
    global AX, T
    plt.close(fig)
    plt.figure(fig, figsize=(8, 5))
    S = subp(len(fit["track"]))
    # print(len(fit['track']), S)
    fontsize = min(max(12 / np.sqrt(S[1]), 5), 10)

    # -- plot chi2
    k = "reduced chi2"
    AX = {k: plt.subplot(S[1], S[0], 1)}
    T = fit["track"]
    plt.plot(fit["track"][k], ".-g")
    # plt.ylabel(k, fontsize=fontsize)
    plt.title(
        k,
        fontsize=fontsize,
        x=0.02,
        y=0.9,
        ha="left",
        va="top",
        bbox={"color": "w", "alpha": 0.1},
    )
    if fit["track"][k][0] / fit["track"][k][1] > 10:
        plt.yscale("log")
    if S[0] > 1:
        AX[k].xaxis.set_visible(False)
    plt.yticks(fontsize=fontsize)
    plt.hlines(fit["chi2"], 0, len(fit["track"][k]) - 1, alpha=0.5, color="orange")
    if fit["track"][k][0] / fit["track"][k][-1] > 100:
        AX[k].set_yscale("log")

    # -- plot all parameters:
    for i, k in enumerate(sorted(filter(lambda x: x != "reduced chi2", fit["track"].keys()))):
        r = np.arange(len(fit["track"][k]))

        AX[k] = plt.subplot(S[1], S[0], i + 2, sharex=AX["reduced chi2"])

        # -- evolution of parameters
        if k in fit["not significant"] or k in fit["not converging"]:
            color = (0.8, 0.3, 0)
        else:
            color = (0, 0.4, 0.8)
        plt.plot(r, fit["track"][k], "-", color=color)
        # -- when parameter converged within uncertainty
        w = np.abs(fit["track"][k] - fit["best"][k]) <= fit["uncer"][k]
        # -- from the end of the sequence, when parameter was within error
        r0 = np.argmax([all(w[x:]) for x in range(len(w))])

        # r0 = r[w][::-1][np.argmax((np.diff(r[w])!=1)[::-1])]
        # plt.plot(r[w], fit['track'][k][w], 'o', color=color)

        plt.plot(r[r >= r0], fit["track"][k][r >= r0], ".", color=color)
        plt.title(k, fontsize=fontsize, x=0.05, y=0.9, ha="left", va="top")

        plt.yticks(fontsize=fontsize)
        if i + 2 < len(fit["track"]) - S[0] + 1:
            AX[k].xaxis.set_visible(False)
        else:
            plt.xticks(fontsize=fontsize)
        plt.fill_between(
            [0, len(fit["track"][k]) - 1],
            fit["best"][k] - fit["uncer"][k],
            fit["best"][k] + fit["uncer"][k],
            color="orange",
            alpha=0.2,
        )
        plt.hlines(fit["best"][k], 0, len(fit["track"][k]) - 1, alpha=0.5, color="orange")
        plt.vlines(r0, plt.ylim()[0], plt.ylim()[1], linestyle=":", color="k", alpha=0.5)
    for k in AX:
        AX[k].callbacks.connect("xlim_changed", _callbackAxes)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
