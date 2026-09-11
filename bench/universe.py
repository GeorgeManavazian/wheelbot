"""Live trading universe: price-diverse, liquid, OPTIONABLE. Spans price tiers so
the bot is testable across $5k-$500k accounts (one put ties up strike*100
collateral, so small accounts can only trade cheap underlyings). Owner-editable.

Sourced from the S&P 500 + Nasdaq 100 + liquid ETFs + a low-price cohort for
small-account testing. Curated for LIQUID LISTED OPTIONS: no mutual funds, no
dotted symbols (Schwab can't route "BRK.B"). A delisted/renamed ticker degrades
gracefully -- LiveMarket skips it and logs it (see market_live.skipped) -- so a
stale entry costs nothing but a logged skip. Re-vet quarterly for delistings."""

# Liquid index / sector / commodity / bond ETFs.
_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "VOO", "IVV", "VTI", "VEA", "VWO", "EFA",
    "EEM", "FXI", "EWZ", "GLD", "SLV", "GDX", "GDXJ", "USO", "XOP", "XLE",
    "XLF", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB", "XLY", "XLRE", "XLC",
    "SMH", "SOXL", "XBI", "KRE", "ARKK", "TLT", "HYG", "LQD", "AGG", "IEF",
    "VNQ", "IBIT", "KWEB",
]

# Low-price cohort ($5-50) -- the inventory a ~$5k account can trade.
_CHEAP = [
    "SOFI", "NIO", "RIG", "SNAP", "NOK", "SIRI", "PLUG", "AG", "HL", "MARA",
    "RIOT", "CLSK", "HOOD", "LYFT", "PBR", "KGC", "AGNC", "VALE", "BITO", "AA",
    "F", "T", "BAC", "INTC", "CHPT", "GPRO", "UPST", "CCL", "CVS", "PARA",
    "KVUE",
]

# S&P 500 / large-cap, A-M.
_SP_AM = [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
    "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK",
    "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT",
    "APP", "APTV", "ACGL", "ADM", "ARES", "ANET", "AJG", "AIZ", "T", "ATO",
    "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX",
    "BDX", "BBY", "TECH", "BIIB", "BLK", "BX", "XYZ", "BNY", "BA", "BKNG",
    "BSX", "BMY", "AVGO", "BR", "BRO", "BLDR", "BG", "BXP", "CHRW", "CDNS",
    "CPT", "COF", "CAH", "CCL", "CARR", "CVNA", "CASY", "CAT", "CBOE", "CBRE",
    "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG",
    "CB", "CHD", "CIEN", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX",
    "CME", "CMS", "KO", "CTSH", "COHR", "COIN", "CL", "CMCSA", "FIX", "COP",
    "ED", "STZ", "CEG", "COO", "CPRT", "GLW", "CPAY", "CTVA", "CSGP", "COST",
    "CRH", "CRWD", "CCI", "CSX", "CMI", "CVS", "DHR", "DRI", "DDOG", "DVA",
    "DECK", "DE", "DELL", "DAL", "DVN", "DXCM", "FANG", "DLR", "DG", "DLTR",
    "D", "DPZ", "DASH", "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "ETN",
    "EBAY", "ECL", "EIX", "EW", "EA", "ELV", "EME", "EMR", "ETR", "EOG",
    "EQT", "EFX", "EQIX", "EQR", "ESS", "EL", "EG", "EVRG", "ES", "EXC",
    "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX",
    "FIS", "FITB", "FSLR", "FE", "FI", "FLEX", "F", "FTNT", "FTV", "FOXA",
    "FOX", "BEN", "FCX", "GRMN", "IT", "GE", "GEHC", "GEV", "GEN", "GNRC",
    "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL", "GDDY", "GS", "HAL",
    "HIG", "HAS", "HCA", "DOC", "HSIC", "HSY", "HPE", "HLT", "HD", "HON",
    "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX",
    "IDXX", "ITW", "INCY", "IR", "PODD", "INTC", "IBKR", "ICE", "IFF", "IP",
    "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J",
    "JNJ", "JCI", "JPM", "KVUE", "KDP", "KEY", "KEYS", "KMB", "KIM", "KMI",
    "KKR", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX", "LVS", "LDOS", "LEN",
    "LII", "LLY", "LIN", "LYV", "LMT", "L", "LOW", "LULU", "LITE", "LYB",
    "MTB", "MPC", "MAR", "MLM", "MRVL", "MAS", "MA", "MKC", "MCD", "MCK",
    "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA",
    "MRNA", "TAP", "MDLZ", "MPWR", "MNST", "MCO",
]

# S&P 500 / large-cap, N-Z.
_SP_NZ = [
    "NDAQ", "NTAP", "NFLX", "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN",
    "NOC", "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL",
    "OMC", "ON", "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PLTR", "PANW", "PARA",
    "PH", "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX",
    "PNW", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU",
    "PEG", "PTC", "PSA", "PHM", "QCOM", "PWR", "DGX", "RL", "RJF", "RTX",
    "O", "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "ROK", "ROL", "ROP",
    "ROST", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", "SRE", "NOW", "SHW",
    "SPG", "SWKS", "SJM", "SW", "SNA", "SOLV", "SO", "LUV", "SWK", "SBUX",
    "STT", "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS", "SYY", "TMUS", "TROW",
    "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TER", "TSLA", "TXN", "TXT",
    "TMO", "TJX", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN",
    "USB", "UBER", "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", "UNH", "UHS",
    "VLO", "VTR", "VLTO", "VRSN", "VRSK", "VZ", "VRTX", "VTRS", "VICI", "V",
    "VST", "VMC", "WAB", "WBA", "WMT", "DIS", "WBD", "WM", "WAT", "WEC",
    "WFC", "WELL", "WST", "WDC", "WY", "WMB", "WTW", "WDAY", "WYNN", "XEL",
    "XYL", "YUM", "ZBRA", "ZBH", "ZTS",
]

# First-seen dedup across the groups; order is stable (ETFs, then cheap, then S&P).
_ALL = _ETFS + _CHEAP + _SP_AM + _SP_NZ

# ...then subtract the reserved one-shot tickers. These are held back as a
# genuinely unseen final read, and trading one spends that read permanently.
# The engine's own refusal (portfolio.RESERVED_TICKERS) guards ONLY the 9-ticker
# default path; the live bot hands LiveMarket an explicit 547-name universe, so
# until 2026-07-31 nothing stopped them at all. They were not hypothetically at
# risk: on 2026-07-20 the bot traded RIG at rank #20 with ARKK at #19, and on
# 2026-07-31 EWZ ranked #7 of 12 gate passers at a price every account can
# afford. Subtract here rather than delete from the lists above, so the lists
# stay a faithful record of the source indices and the exclusion states itself.
from src.engine_v2.options.portfolio import RESERVED_TICKERS

# B6 re-vet (owner kill-list, 2026-08-02 -- all 11 approved). Same
# subtract-don't-delete idiom as the reserved set: the source lists above stay
# a faithful record of the indices they came from, and the ban states itself.
# Dead / ghost / penny: PARA became PSKY 2025-08-07 yet the vendor still
# serves its frozen $11.04 close, which RANKS NORMALLY -- the silent-staleness
# class the retirement exists to remove; WBA went private (zero local data);
# GPRO/PLUG are sub-$1 with near-empty books; CHPT is a 1:20-reverse-split
# death spiral. Surgery-scarred (mergers/spinoffs/symbol-reuse with
# unadjusted history and thin books): SIRI, AMCR, DD, DOC, FLEX, FTV.
# Evidence: the B6 sweep kill-list, 2026-08-02.
_RETIRED = frozenset({
    "PARA", "WBA", "GPRO", "CHPT", "PLUG",
    "SIRI", "AMCR", "DD", "DOC", "FLEX", "FTV",
})

# Liquidity/yield screen (2026-08-27). Measured on every near-the-money PUT
# (|delta| 0.2-0.4, DTE 7-15 -- the shape this bot actually sells) across the
# full OI-clean live window (2024-08 -> 2026-07): a name whose MEDIAN
# annualized yield on collateral (credit/strike * 365/dte, priced at bid,
# same formula as the live `min_ann_yield_on_collateral=0.08` floor) never
# clears 0.08 will fail that floor on a typical day, not an unlucky one.
# These are low-volatility utilities/staples/REITs/bond-ETFs where a put
# simply doesn't pay enough to justify the strike*100 collateral -- not a
# liquidity problem, a capital-efficiency one. Confirmed against the
# promoted config's 142-campaign live fill log: 36 of 38 names below the
# floor never traded once in 23 months. The other 2 (WBD, KRE) did trade
# and are kept. Re-run the screen before trusting this list against a
# materially different config.
_LOW_YIELD = frozenset({
    "EFA", "XLE", "XLV", "XLI", "XLP", "XLU", "XLB", "XLY", "XLRE", "XLC",
    "HYG", "LQD", "AGG", "IEF",
    "KVUE", "ALLE", "LNT", "AIZ", "ATO", "CPT", "CMS", "CPAY", "ESS",
    "EVRG", "FRT", "FOXA", "FOX", "HSIC", "IEX", "JKHY", "NWSA", "NWS",
    "REG", "TDY", "UDR", "VLTO",
})

UNIVERSE = [t for t in dict.fromkeys(_ALL)
            if t not in RESERVED_TICKERS and t not in _RETIRED
            and t not in _LOW_YIELD]
