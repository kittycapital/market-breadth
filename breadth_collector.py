"""
Market Breadth Collector for Herdvibe.com
=========================================
S&P 500 / Nasdaq 100 / Dow 30 구성종목의 % Above Moving Average를 계산합니다.
yfinance를 사용하여 데이터를 수집하고, JSON 파일로 저장합니다.

사용법:
  - 초기 구축: python breadth_collector.py --init
  - 일일 업데이트: python breadth_collector.py
  - 크론잡 설정 (미국장 마감 후, 한국시간 오전 6시):
    0 6 * * 2-6 cd /path/to/project && python breadth_collector.py

필요 패키지: pip install yfinance pandas
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 인덱스 구성종목 정의
# ============================================================

SP500_TICKERS = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB",
    "ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AEP","AXP",
    "AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","AON","APA","APO","AAPL","AMAT","APP",
    "APTV","ACGL","ADM","ARES","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY",
    "AXON","BKR","BALL","BAC","BAX","BDX","BRK-B","BBY","TECH","BIIB","BLK","BX","XYZ","BK",
    "BA","BKNG","BSX","BMY","AVGO","BR","BRO","BF-B","BLDR","BG","BXP","CHRW","CDNS","CPT",
    "CPB","COF","CAH","CCL","CARR","CVNA","CAT","CBOE","CBRE","CDW","COR","CNC","CNP","CF",
    "CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CIEN","CI","CINF","CTAS","CSCO","C","CFG",
    "CLX","CME","CMS","KO","CTSH","COIN","CL","CMCSA","FIX","CAG","COP","ED","STZ","CEG",
    "COO","CPRT","GLW","CPAY","CTVA","CSGP","COST","CTRA","CRH","CRWD","CCI","CSX","CMI",
    "CVS","DHR","DRI","DDOG","DVA","DECK","DE","DELL","DAL","DVN","DXCM","FANG","DLR","DG",
    "DLTR","D","DPZ","DASH","DOV","DOW","DHI","DTE","DUK","DD","ETN","EBAY","ECL","EIX","EW",
    "EA","ELV","EME","EMR","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ERIE","ESS","EL",
    "EG","EVRG","ES","EXC","EXE","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT",
    "FDX","FIS","FITB","FSLR","FE","FISV","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN",
    "IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GPN","GL","GDDY","GS",
    "HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HPE","HLT","HOLX","HD","HON","HRL","HST",
    "HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC",
    "IBKR","ICE","IFF","IP","INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J",
    "JNJ","JCI","JPM","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KKR","KLAC","KHC","KR",
    "LHX","LH","LRCX","LW","LVS","LDOS","LEN","LII","LLY","LIN","LYV","LMT","L","LOW",
    "LULU","LYB","MTB","MPC","MAR","MRSH","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT",
    "MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MOH","TAP","MDLZ","MPWR",
    "MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE",
    "NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL",
    "OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PLTR","PANW","PSKY","PH","PAYX","PAYC","PYPL",
    "PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD",
    "PRU","PEG","PTC","PSA","PHM","PWR","QCOM","DGX","Q","RL","RJF","RTX","O","REG","REGN",
    "RF","RSG","RMD","RVTY","HOOD","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SNDK","SBAC",
    "SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV","SWK",
    "SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR",
    "TRGP","TGT","TEL","TDY","TER","TSLA","TXN","TPL","TXT","TMO","TJX","TKO","TSCO","TT",
    "TDG","TRV","TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI",
    "UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC",
    "WRB","GWW","WAB","WMT","DIS","WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WSM",
    "WMB","WTW","WDAY","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS"
]

NASDAQ100_TICKERS = [
    "AAPL","ABNB","ADBE","ADI","ADP","ADSK","AEP","ALNY","AMAT","AMD","AMGN","AMZN","APP",
    "ARM","ASML","AVGO","AXON","BKR","BKNG","CCEP","CDNS","CEG","CHTR","CMCSA","COST","CPRT",
    "CRWD","CSGP","CSCO","CSX","CTAS","CTSH","DASH","DDOG","DXCM","EA","EXC","FANG","FAST",
    "FER","FTNT","GEHC","GILD","GOOG","GOOGL","HON","IDXX","INSM","INTC","INTU","ISRG","KDP",
    "KHC","KLAC","LIN","LRCX","MAR","MCHP","MDLZ","MELI","META","MNST","MPWR","MRVL","MSFT",
    "MSTR","MU","NFLX","NVDA","NXPI","ODFL","ORLY","PANW","PAYX","PCAR","PDD","PEP","PLTR",
    "PYPL","QCOM","REGN","ROP","ROST","SBUX","SHOP","SNPS","SOLS","STX","TEAM","TMUS","TRI",
    "TSLA","TXN","VRTX","WBD","WDAY","WDC","WMT","XEL"
]

DOW30_TICKERS = [
    "UNH","MSFT","GS","HD","MCD","CAT","AMGN","V","CRM","BA","AMZN","HON","AAPL","AXP",
    "JPM","TRV","IBM","JNJ","WMT","CVX","PG","MRK","MMM","DIS","NKE","KO","CSCO","NVDA",
    "VZ","INTC"
]

INDEX_CONFIG = {
    "SPY": {"name": "S&P 500", "tickers": SP500_TICKERS},
    "QQQ": {"name": "Nasdaq 100", "tickers": NASDAQ100_TICKERS},
    "DIA": {"name": "Dow 30", "tickers": DOW30_TICKERS},
}

# 인덱스 ETF 티커 (가격 차트용)
INDEX_ETF_TICKERS = ["SPY", "QQQ", "DIA"]

MA_PERIODS = [20, 50, 100, 200]

OUTPUT_DIR = Path("./data")
OUTPUT_FILE = OUTPUT_DIR / "market_breadth.json"

# ============================================================
# 데이터 수집 함수
# ============================================================

def fetch_prices_batch(tickers, period="5y", batch_size=50):
    """티커를 배치로 나누어 가격 데이터를 가져옵니다."""
    all_data = {}
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"  배치 {batch_num}/{total_batches}: {len(batch)}개 티커 다운로드 중...")

        try:
            data = yf.download(
                batch,
                period=period,
                auto_adjust=True,
                threads=True,
                progress=False
            )

            if isinstance(data.columns, pd.MultiIndex):
                close_data = data["Close"]
            else:
                # 단일 티커인 경우
                close_data = pd.DataFrame(data["Close"], columns=[batch[0]])

            for ticker in batch:
                if ticker in close_data.columns:
                    series = close_data[ticker].dropna()
                    if len(series) > 0:
                        all_data[ticker] = series

        except Exception as e:
            logger.warning(f"  배치 {batch_num} 에러: {e}")

        # rate limit 방지
        if batch_num < total_batches:
            time.sleep(1)

    return all_data


def calculate_breadth(price_dict, ma_periods=MA_PERIODS):
    """
    각 날짜별로 MA 위에 있는 종목 비율을 계산합니다.
    Returns: DataFrame with columns like 'pct_above_20', 'pct_above_50', etc.
    """
    if not price_dict:
        return pd.DataFrame()

    # 모든 종목의 종가를 하나의 DataFrame으로
    close_df = pd.DataFrame(price_dict)

    results = {}

    for period in ma_periods:
        ma_df = close_df.rolling(window=period).mean()
        above_ma = (close_df > ma_df).astype(int)
        # 각 날짜별 유효 종목 수 대비 MA 위 비율
        count_above = above_ma.sum(axis=1)
        count_valid = above_ma.count(axis=1)
        pct_above = (count_above / count_valid * 100).round(2)
        results[f"pct_above_{period}"] = pct_above

    result_df = pd.DataFrame(results)
    # MA 계산에 필요한 최소 기간 이후의 데이터만
    result_df = result_df.dropna()

    return result_df


def fetch_index_prices(period="5y"):
    """인덱스 ETF(SPY, QQQ, DIA) 가격 데이터를 가져옵니다."""
    logger.info("인덱스 ETF 가격 다운로드 중...")
    data = yf.download(INDEX_ETF_TICKERS, period=period, auto_adjust=True, progress=False)

    result = {}
    if isinstance(data.columns, pd.MultiIndex):
        close_data = data["Close"]
    else:
        close_data = pd.DataFrame(data["Close"], columns=[INDEX_ETF_TICKERS[0]])

    for ticker in INDEX_ETF_TICKERS:
        if ticker in close_data.columns:
            series = close_data[ticker].dropna()
            result[ticker] = series

    return result


def fetch_vix(period="5y"):
    """VIX 데이터를 가져옵니다."""
    logger.info("VIX 데이터 다운로드 중...")
    try:
        vix = yf.download("^VIX", period=period, auto_adjust=True, progress=False)
        return vix["Close"].dropna()
    except Exception as e:
        logger.warning(f"VIX 다운로드 실패: {e}")
        return pd.Series()


def calculate_signals(breadth_df, vix_series=None):
    """
    탑/바텀 시그널 스코어를 계산합니다.
    Returns: DataFrame with 'bottom_score', 'top_score' columns
    """
    scores = pd.DataFrame(index=breadth_df.index)

    # === 바텀 스코어 ===
    bottom = pd.Series(0, index=breadth_df.index, dtype=float)

    if "pct_above_20" in breadth_df.columns:
        bottom += (breadth_df["pct_above_20"] < 15).astype(float) * 25
        # 20MA 반등 시작 감지 (전일 대비 상승)
        ma20_diff = breadth_df["pct_above_20"].diff()
        bottom += ((breadth_df["pct_above_20"] < 25) & (ma20_diff > 0)).astype(float) * 10

    if "pct_above_50" in breadth_df.columns:
        bottom += (breadth_df["pct_above_50"] < 25).astype(float) * 20

    if "pct_above_100" in breadth_df.columns:
        bottom += (breadth_df["pct_above_100"] < 35).astype(float) * 15

    if "pct_above_200" in breadth_df.columns:
        bottom += (breadth_df["pct_above_200"] < 45).astype(float) * 10

    # VIX 가산점
    if vix_series is not None and len(vix_series) > 0:
        # vix_series를 1D Series로 확실히 변환
        if isinstance(vix_series, pd.DataFrame):
            vix_s = vix_series.iloc[:, 0]
        else:
            vix_s = vix_series.copy()
        vix_s.index = pd.DatetimeIndex(vix_s.index).normalize()
        breadth_idx = pd.DatetimeIndex(breadth_df.index).normalize()
        vix_aligned = vix_s.reindex(breadth_idx, method="ffill").fillna(0).values
        bottom += ((vix_aligned >= 25) & (vix_aligned < 30)).astype(float) * 5
        bottom += ((vix_aligned >= 30) & (vix_aligned < 35)).astype(float) * 10
        bottom += ((vix_aligned >= 35) & (vix_aligned < 45)).astype(float) * 15
        bottom += (vix_aligned >= 45).astype(float) * 20

    scores["bottom_score"] = bottom.clip(0, 100)

    # === 탑 스코어 ===
    top = pd.Series(0, index=breadth_df.index, dtype=float)

    if "pct_above_20" in breadth_df.columns:
        top += (breadth_df["pct_above_20"] > 85).astype(float) * 20
        # 과열 후 꺾임 감지
        ma20_diff = breadth_df["pct_above_20"].diff()
        top += ((breadth_df["pct_above_20"] > 75) & (ma20_diff < -2)).astype(float) * 15

    if "pct_above_50" in breadth_df.columns:
        top += (breadth_df["pct_above_50"] > 80).astype(float) * 15

    if "pct_above_100" in breadth_df.columns:
        top += (breadth_df["pct_above_100"] > 80).astype(float) * 15

    if "pct_above_200" in breadth_df.columns:
        top += (breadth_df["pct_above_200"] > 85).astype(float) * 10

    # VIX 극단적 저점 (안일함)
    if vix_series is not None and len(vix_series) > 0:
        if isinstance(vix_series, pd.DataFrame):
            vix_s = vix_series.iloc[:, 0]
        else:
            vix_s = vix_series.copy()
        vix_s.index = pd.DatetimeIndex(vix_s.index).normalize()
        breadth_idx = pd.DatetimeIndex(breadth_df.index).normalize()
        vix_aligned = vix_s.reindex(breadth_idx, method="ffill").fillna(20).values
        top += ((vix_aligned <= 14) & (vix_aligned > 12)).astype(float) * 5
        top += (vix_aligned <= 12).astype(float) * 10

    scores["top_score"] = top.clip(0, 100)

    return scores


def calculate_historical_probability(breadth_df, index_prices, index_key):
    """
    현재와 유사한 브레드스 구간에서의 미래 수익률을 계산합니다.
    동적 범위: ±3% → ±5% → ±8%
    """
    if index_key not in index_prices or breadth_df.empty:
        return None

    prices = index_prices[index_key]
    # 브레드스와 가격의 공통 날짜
    common_dates = breadth_df.index.intersection(prices.index)
    if len(common_dates) < 100:
        return None

    current_20 = breadth_df.loc[common_dates[-1], "pct_above_20"]
    current_50 = breadth_df.loc[common_dates[-1], "pct_above_50"]

    # 동적 범위로 유사 구간 검색
    forward_returns = {5: [], 20: [], 60: []}  # 1주, 1개월, 3개월

    for tolerance in [3, 5, 8]:
        matches = common_dates[
            (abs(breadth_df.loc[common_dates, "pct_above_20"] - current_20) <= tolerance) &
            (abs(breadth_df.loc[common_dates, "pct_above_50"] - current_50) <= tolerance)
        ]
        # 최근 20일은 제외 (자기 자신 방지)
        matches = matches[matches < common_dates[-20]]

        if len(matches) >= 5:
            for match_date in matches:
                idx_pos = common_dates.get_loc(match_date)
                for days, label in [(5, 5), (20, 20), (60, 60)]:
                    if idx_pos + days < len(common_dates):
                        future_date = common_dates[idx_pos + days]
                        ret = (prices.loc[future_date] / prices.loc[match_date] - 1) * 100
                        forward_returns[label].append(float(ret))
            break

    if not forward_returns[5]:
        return None

    result = {"tolerance": tolerance, "match_count": len(matches)}
    for days in [5, 20, 60]:
        if forward_returns[days]:
            arr = np.array(forward_returns[days])
            result[f"{days}d"] = {
                "avg": round(float(np.mean(arr)), 2),
                "positive_pct": round(float(np.sum(arr > 0) / len(arr) * 100), 1),
                "max": round(float(np.max(arr)), 2),
                "min": round(float(np.min(arr)), 2),
            }

    return result


def build_output(index_breadth, index_prices, vix_series, signals_dict):
    """최종 JSON 출력 데이터를 구성합니다."""
    output = {
        "generated_at": datetime.now().isoformat(),
        "indices": {}
    }

    for key in ["SPY", "QQQ", "DIA"]:
        breadth_df = index_breadth.get(key)
        if breadth_df is None or breadth_df.empty:
            continue

        # 날짜를 문자열로 변환
        dates = [d.strftime("%Y-%m-%d") for d in breadth_df.index]

        index_data = {
            "name": INDEX_CONFIG[key]["name"],
            "dates": dates,
            "breadth": {
                f"pct_above_{p}": breadth_df[f"pct_above_{p}"].tolist()
                for p in MA_PERIODS
                if f"pct_above_{p}" in breadth_df.columns
            },
        }

        # 인덱스 가격
        if key in index_prices:
            price_series = index_prices[key].reindex(breadth_df.index, method="ffill")
            index_data["prices"] = [round(float(p), 2) for p in price_series.values]

        # 시그널 스코어
        if key in signals_dict:
            sig = signals_dict[key]
            index_data["signals"] = {
                "bottom_score": sig["bottom_score"].tolist(),
                "top_score": sig["top_score"].tolist(),
            }

        # 현재 스냅샷 (최신 값)
        latest = breadth_df.iloc[-1]
        index_data["current"] = {
            f"pct_above_{p}": round(float(latest[f"pct_above_{p}"]), 1)
            for p in MA_PERIODS
            if f"pct_above_{p}" in latest.index
        }

        if key in signals_dict:
            sig_latest = signals_dict[key].iloc[-1]
            index_data["current"]["bottom_score"] = round(float(sig_latest["bottom_score"]), 0)
            index_data["current"]["top_score"] = round(float(sig_latest["top_score"]), 0)

        # 히스토리컬 확률
        hist_prob = calculate_historical_probability(breadth_df, index_prices, key)
        if hist_prob:
            index_data["historical_probability"] = hist_prob

        output["indices"][key] = index_data

    # VIX
    if vix_series is not None and len(vix_series) > 0:
        output["vix"] = {
            "current": round(float(vix_series.iloc[-1]), 2),
            "dates": [d.strftime("%Y-%m-%d") for d in vix_series.index],
            "values": [round(float(v), 2) for v in vix_series.values],
        }

    return output


# ============================================================
# 메인 실행
# ============================================================

def main(init_mode=False):
    """
    init_mode=True: 5년 전체 데이터를 처음부터 수집
    init_mode=False: 최근 5일 데이터로 업데이트 (일일 크론잡)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    period = "5y" if init_mode else "5y"  # 항상 5년 (MA200 계산에 충분한 데이터 필요)
    logger.info(f"=== Market Breadth Collector 시작 (mode={'init' if init_mode else 'update'}) ===")

    # 1. 인덱스 ETF 가격 수집
    index_prices = fetch_index_prices(period)
    logger.info(f"인덱스 가격 수집 완료: {list(index_prices.keys())}")

    # 2. VIX 수집
    vix_series = fetch_vix(period)
    logger.info(f"VIX 수집 완료: {len(vix_series)}일")

    # 3. 각 인덱스별 브레드스 계산
    index_breadth = {}
    signals_dict = {}

    for key, config in INDEX_CONFIG.items():
        logger.info(f"\n--- {config['name']} ({key}) ---")
        tickers = config["tickers"]
        logger.info(f"종목 수: {len(tickers)}")

        # 가격 수집
        price_dict = fetch_prices_batch(tickers, period=period)
        logger.info(f"수집 성공: {len(price_dict)}/{len(tickers)} 종목")

        # 실패 티커 기록
        failed = set(tickers) - set(price_dict.keys())
        if failed:
            logger.warning(f"수집 실패 티커: {failed}")

        # 브레드스 계산
        breadth_df = calculate_breadth(price_dict)
        if not breadth_df.empty:
            index_breadth[key] = breadth_df
            logger.info(f"브레드스 계산 완료: {len(breadth_df)}일, "
                       f"최신 날짜: {breadth_df.index[-1].strftime('%Y-%m-%d')}")

            # 시그널 스코어 계산
            signals = calculate_signals(breadth_df, vix_series)
            signals_dict[key] = signals

            # 현재 상태 출력
            latest = breadth_df.iloc[-1]
            for p in MA_PERIODS:
                col = f"pct_above_{p}"
                if col in latest.index:
                    logger.info(f"  > {p}MA 위: {latest[col]:.1f}%")

    # 4. JSON 출력
    output = build_output(index_breadth, index_prices, vix_series, signals_dict)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    logger.info(f"\n=== 완료! ===")
    logger.info(f"출력 파일: {OUTPUT_FILE} ({file_size:.1f}MB)")
    logger.info(f"생성 시각: {output['generated_at']}")


if __name__ == "__main__":
    import sys
    init_mode = "--init" in sys.argv
    main(init_mode=init_mode)
