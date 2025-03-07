import asyncio
import json
import os
import re
import ssl

import aiohttp
import certifi
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from dotenv import load_dotenv
from subgrounds import Subgrounds
import streamlit as st
from allora_sdk.v2.api_client import (
    AlloraAPIClient,
    ChainSlug
)

from src.meowfi.constants import (
    CAMELOT_QUERY_MAPPING,
    RISK_PARAMS, STABLE_COINS_CAMELOT, STABLE_COINS_AERODROME,
    ETHEREUM_COINS_AERODROME, OTHER_COINS_AERODROME, OTHER_COINS_CAMELOT,
    ETHEREUM_COINS_CAMELOT, VALID_CHAINS, ALLORA_TOPIC_ID_MAP,
)

load_dotenv()
SUBGRAPH_API_KEY = os.getenv('SUBGRAPH_API_KEY')

ssl_context = ssl.create_default_context(cafile=certifi.where())

import warnings

warnings.simplefilter("ignore", ResourceWarning)


def get_valid_chains():
    try:
        return st.session_state.valid_chains
    except:
        return VALID_CHAINS


def correct_symbol_case(symbol, token_lists):
    symbol_lower = symbol.lower()
    for token_list in token_lists:
        for token in token_list:
            if token.lower() == symbol_lower:
                return token
    return symbol


def clean_token_symbols(token0_filter, token1_filter):
    token0_symbols = set(token0_filter)  # Convert token0 list to a set
    token1_symbols = set(token1_filter)  # Convert token1 list to a set

    # Remove duplicates from token1
    cleaned_token1 = list(token1_symbols - token0_symbols)

    return cleaned_token1


def fetch_pools(pools_filters, limit=10):
    """
    Fetch liquidity pool data from blockchain subgraphs based on filters.

    If the first query returns an empty DataFrame, swap token positions (token0 ↔ token1)
    and try fetching again.

    Args:
        pools_filters (dict): Filters including chain, pool type, TVL sorting, symbol, and risk.
        limit (int): Maximum number of pools to return per chain.

    Returns:
        pd.DataFrame: Combined pool data from specified chains.
    """
    try:
        with Subgrounds() as sg:
            chain = pools_filters.get("chain", "all").lower()
            order_by = "totalValueLockedUSD"
            order_direction = "desc" if pools_filters.get("tvlUSD", "high") == "high" else "asc"
            risk_level = pools_filters.get("risk", "medium").lower()
            pool_type = pools_filters.get("pool_type", "stablecoin").lower()
            symbol_filter = pools_filters.get("symbol", "").upper()

            # Ensure symbol case correction

            # Assign correct symbols
            symbol_filter = correct_symbol_case(symbol_filter,
                                                [STABLE_COINS_CAMELOT, ETHEREUM_COINS_CAMELOT, OTHER_COINS_CAMELOT,
                                                 STABLE_COINS_AERODROME, ETHEREUM_COINS_AERODROME,
                                                 OTHER_COINS_AERODROME])

            # Special case: ETH on Arbitrum should be WETH
            if symbol_filter == "ETH":
                symbol_filter = "WETH"

            # Setup chain-specific coin lists and TVL thresholds
            if chain.startswith("arb"):
                STABLE_COINS = STABLE_COINS_CAMELOT
                ETHEREUM_COINS = ETHEREUM_COINS_CAMELOT
                OTHER_COINS = OTHER_COINS_CAMELOT
                tvl_threshold = 500000
            elif chain.startswith("base"):
                STABLE_COINS = STABLE_COINS_AERODROME
                ETHEREUM_COINS = ETHEREUM_COINS_AERODROME
                OTHER_COINS = OTHER_COINS_AERODROME
                tvl_threshold = 1000000
            else:
                STABLE_COINS = STABLE_COINS_CAMELOT + STABLE_COINS_AERODROME
                ETHEREUM_COINS = ETHEREUM_COINS_CAMELOT + ETHEREUM_COINS_AERODROME
                OTHER_COINS = OTHER_COINS_CAMELOT + OTHER_COINS_AERODROME
                tvl_threshold = 1000000

            # Determine token categories
            if symbol_filter:
                token1_filter = [symbol_filter]
                if symbol_filter == "BTC":
                    token1_filter = ["WBTC", "cbBTC", "tBTC", "LBTC"]

                if symbol_filter in ETHEREUM_COINS:
                    token_category = "ethereum"
                elif symbol_filter in STABLE_COINS:
                    token_category = "stablecoin"
                elif symbol_filter in OTHER_COINS:
                    token_category = "other"
                else:
                    token_category = None

                # Set token2_filter based on risk and category
                if risk_level == "low":
                    token2_filter = STABLE_COINS if token_category == "stablecoin" else ETHEREUM_COINS if token_category == "ethereum" else OTHER_COINS
                    range_type = "wide"
                elif risk_level == "high":
                    if token_category == "ethereum":
                        token2_filter = STABLE_COINS + OTHER_COINS
                    elif token_category == "stablecoin":
                        token2_filter = ETHEREUM_COINS + OTHER_COINS
                    else:
                        token2_filter = STABLE_COINS + ETHEREUM_COINS
                    range_type = "narrow"
                else:

                    token2_filter = STABLE_COINS + ETHEREUM_COINS + OTHER_COINS
                    if not symbol_filter and risk_level == "medium":
                        token2_filter = clean_token_symbols(token0_filter=token1_filter, token1_filter=token2_filter)
                    range_type = "medium"
            else:
                if pool_type == "stablecoin":
                    token1_filter = STABLE_COINS
                elif pool_type == "bluechip":
                    token1_filter = ETHEREUM_COINS
                else:
                    token1_filter = OTHER_COINS

                if risk_level == "low":
                    token2_filter = token1_filter
                    range_type = "wide"
                elif risk_level == "high":
                    if pool_type == "other":
                        token2_filter = STABLE_COINS + ETHEREUM_COINS
                    else:
                        token2_filter = OTHER_COINS
                    range_type = "narrow"
                else:
                    token2_filter = STABLE_COINS + ETHEREUM_COINS + OTHER_COINS
                    if not symbol_filter and risk_level == "medium":
                        token2_filter = clean_token_symbols(token0_filter=token1_filter, token1_filter=token2_filter)
                    range_type = "medium"

            # Special handling for Aerodrome on Base
            if chain.startswith("base"):
                pool_type = "Concentrated Volatile"

            where_conditions = {
                "totalValueLockedUSD_gt": tvl_threshold,
                "token0_": {"symbol_in": token1_filter},
                "token1_": {"symbol_in": token2_filter},
            }
            print(where_conditions)
            VALID_CHAINS = get_valid_chains()
            chains_to_fetch = [chain] if chain in VALID_CHAINS else VALID_CHAINS.keys()
            results = []
            for ch in chains_to_fetch:
                subgraph_url = VALID_CHAINS[ch].format(api_key=SUBGRAPH_API_KEY)
                camelot = sg.load_subgraph(subgraph_url)

                pools = camelot.Query.pools(
                    where=where_conditions,
                    first=limit,
                    orderBy=order_by,
                    orderDirection=order_direction,
                )

                df = sg.query_df([
                    pools.id, pools.totalValueLockedUSD, pools.txCount,
                    pools.token0.id, pools.token0.symbol,
                    pools.token1.id, pools.token1.symbol
                ], columns=["pool_id", "tvl_usd", "txn_count", "token0_id", "token0_symbol", "token1_id",
                            "token1_symbol"])

                df["chain"] = ch
                df["range"] = range_type
                df["risk"] = risk_level
                results.append(df)
            print("running 2")

            final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
            swapped_where_conditions = {
                "totalValueLockedUSD_gt": tvl_threshold,
                "token0_": {"symbol_in": token2_filter},
                "token1_": {"symbol_in": token1_filter},
            }

            for ch in chains_to_fetch:
                subgraph_url = VALID_CHAINS[ch].format(api_key=SUBGRAPH_API_KEY)
                camelot = sg.load_subgraph(subgraph_url)

                pools_swapped = camelot.Query.pools(
                    where=swapped_where_conditions,
                    first=limit,
                    orderBy=order_by,
                    orderDirection=order_direction,
                )

                df_swapped = sg.query_df([
                    pools_swapped.id, pools_swapped.totalValueLockedUSD, pools_swapped.txCount,
                    pools_swapped.token0.id, pools_swapped.token0.symbol,
                    pools_swapped.token1.id, pools_swapped.token1.symbol
                ], columns=["pool_id", "tvl_usd", "txn_count", "token0_id", "token0_symbol", "token1_id",
                            "token1_symbol"])

                df_swapped["chain"] = ch
                df_swapped["range"] = range_type
                df_swapped["risk"] = risk_level
                results.append(df_swapped)

                final_df = pd.concat(results, ignore_index=True)
            return final_df
    except Exception as e:
        print(f"ERROR: fetch_pools {e}")
        return "sorry no data found tell this to user"


# pool_filters = {
#     # "pool_type": "bluechip",
#     "tvlUSD": "high",
#     "chain": "arbitrum",
#     # "symbol": "eth",
#     "risk": "medium"
# }
# print(fetch_pools(pools_filters=pool_filters))


# ----------------------------- Price Analysis Metrics -----------------------------

def calculate_volatility(price_data):
    """
    Calculate annualized volatility from historical price data.

    Args:
        price_data (dict): Pool's daily price data from subgraph

    Returns:
        dict: Volatility percentages for both tokens in the pool
    """
    pool_day_data = price_data.get('data', {}).get('pool', {}).get('poolDayData', [])

    if not pool_day_data:
        return {"token0_volatility": 0.0, "token1_volatility": 0.0}

    df = pd.DataFrame(pool_day_data)

    # Ensure price columns exist and convert to float
    if 'token0Price' not in df or 'token1Price' not in df:
        return {"token0_volatility": 0.0, "token1_volatility": 0.0}

    df['token0Price'] = pd.to_numeric(df['token0Price'], errors='coerce')
    df['token1Price'] = pd.to_numeric(df['token1Price'], errors='coerce')

    # Drop NaN values before calculating percentage change
    vol_token0 = df['token0Price'].pct_change().dropna().std() * np.sqrt(365)
    vol_token1 = df['token1Price'].pct_change().dropna().std() * np.sqrt(365)

    # Handle NaN cases
    vol_token0 = round(vol_token0, 4) if pd.notna(vol_token0) else 0.0
    vol_token1 = round(vol_token1, 4) if pd.notna(vol_token1) else 0.0

    return {
        "token0_volatility": vol_token0,
        "token1_volatility": vol_token1
    }


def calculate_liquidity_concentration(liquidity_data):
    """
    Calculate liquidity concentration metrics (HHI and Gini Coefficient).

    Args:
        liquidity_data (dict): Pool's liquidity distribution data

    Returns:
        dict: HHI and Gini metrics
    """

    ticks = liquidity_data.get('data', {}).get('pool', {}).get('ticks', [])

    liquidity_values = [abs(int(tick.get('liquidityNet', 0))) for tick in ticks if
                        int(tick.get('liquidityNet', 0)) != 0]

    total_liquidity = sum(liquidity_values)
    if total_liquidity == 0:
        return {"HHI": 0, "Gini Coefficient": 0}

    # Normalize liquidity values
    normalized_liquidity = [liq / total_liquidity for liq in liquidity_values]

    # Calculate Herfindahl-Hirschman Index (HHI)
    hhi = sum(x ** 2 for x in normalized_liquidity)

    # Calculate Gini Coefficient
    sorted_liq = sorted(normalized_liquidity)
    n = len(sorted_liq)

    if n == 0:
        return {"HHI": round(hhi, 4), "Gini Coefficient": 0}

    gini = (2 * sum((i + 1) * liq for i, liq in enumerate(sorted_liq)) / (n * sum(sorted_liq))) - (n + 1) / n

    return {
        "HHI": round(hhi, 4),
        "Gini Coefficient": round(gini, 4)
    }


def calculate_volume_fee_analysis(volume_data):
    """
    Analyze trading volume and fee metrics.

    Args:
        volume_data (dict): Pool's volume and fee data

    Returns:
        dict: Aggregated volume, fees, and TVL metrics
    """
    pool_data = volume_data.get('data', {}).get('pool', {})
    pool_day_data = pool_data.get('poolDayData', [])

    if not pool_day_data:
        return {
            "total_vol_usd": 0.0,
            "total_fees_usd": 0.0,
            "avg_fee_rate": 0.0,
            "avg_tvl_usd": 0.0
        }

    df = pd.DataFrame(pool_day_data)

    if 'feesUSD' not in df or 'volumeUSD' not in df:
        return {
            "total_vol_usd": 0.0,
            "total_fees_usd": 0.0,
            "avg_fee_rate": 0.0,
            "avg_tvl_usd": 0.0
        }

    # Convert necessary fields to float
    df['feesUSD'] = pd.to_numeric(df['feesUSD'], errors='coerce').fillna(0.0)
    df['volumeUSD'] = pd.to_numeric(df['volumeUSD'], errors='coerce').fillna(0.0)

    # Compute required metrics
    total_volume = df['volumeUSD'].sum()
    total_fees = df['feesUSD'].sum()
    avg_fee_rate = (total_fees / total_volume) if total_volume != 0 else 0

    total_value_locked = pool_data.get('totalValueLockedUSD', 0.0)
    average_tvl = float(total_value_locked) if total_value_locked else 0.0
    average_daily_vol = (total_volume / len(df)) if len(df) > 0 else 0.0
    feeZtO = pool_data.get('feeZtO', 0.0)
    fee_tier = int(feeZtO) / 1e6

    return {
        "total_vol_usd": round(total_volume, 2),
        "total_fees_usd": round(total_fees, 2),
        "avg_fee_rate": round(avg_fee_rate, 6),
        "avg_tvl_usd": round(average_tvl, 2),
        "avg_daily_vol_usd": round(average_daily_vol, 2),
        "fee_tier": fee_tier
    }


def calculate_apr(apr_data, tvl, average_tvl):
    """
    Calculate Annual Percentage Return metrics.

    Args:
        apr_data (dict): Fee data from different time ranges
        tvl (float): Current Total Value Locked
        average_tvl (float): Historical average TVL

    Returns:
        dict: Daily and average APR percentages
    """
    daily_fees = float(apr_data.get('data', {}).get('pool', {}).get('daily', [{}])[0].get('feesUSD', 0))
    monthly_fees = sum(
        float(day.get('feesUSD', 0)) for day in apr_data.get('data', {}).get('pool', {}).get('monthly', []))

    print(f"Daily Fees: {daily_fees}, Monthly Fees: {monthly_fees}, TVL: {average_tvl}")

    daily_apr = (daily_fees * 365 * 100) / float(tvl or 1)
    average_apr = (monthly_fees * 12 * 100) / float(average_tvl or 1)

    return {
        "daily_apr": round(daily_apr, 4),
        "average_apr": round(average_apr, 4)
    }


async def analyze_pool(session, pool_id, semaphore, chain, risk, symbol):
    """
    Async wrapper for pool analysis pipeline.

    Args:
        session (aiohttp.ClientSession): HTTP session
        pool_id (str): Pool ID to analyze
        semaphore (asyncio.Semaphore): Concurrency limiter
        chain (str): Blockchain network identifier

    Returns:
        dict: Aggregated analysis results for the pool
    """
    pool_id = pool_id.lower()
    tasks = [
        fetch_data_async(session, pool_id, "price", semaphore, chain),
        fetch_data_async(session, pool_id, "vol", semaphore, chain),
        fetch_data_async(session, pool_id, "liq", semaphore, chain),
        fetch_data_async(session, pool_id, "apr", semaphore, chain),
        fetch_data_async(session, pool_id, "ohlc", semaphore, chain),
        fetch_data_async(session, pool_id, "liq_delta", semaphore, chain)
    ]
    price_data, vol_data, liq_data, apr_data, ohlc_data, liq_delta = await asyncio.gather(*tasks)

    try:

        liq_data = liq_data if isinstance(liq_data, dict) else {}

        price_metadata = price_data.get('data', {}).get('pool', {})
        tvl = float(liq_data.get('data', {}).get('pool', {}).get('totalValueLockedUSD', 1))

        volatility_metrics = calculate_volatility(price_data)
        liquidity_metrics = calculate_liquidity_concentration(liq_data)
        volume_fee_metrics = calculate_volume_fee_analysis(vol_data)
        apr_metrics = calculate_apr(apr_data=apr_data, tvl=tvl, average_tvl=volume_fee_metrics.get('avg_tvl_usd', 0))
        token_symbol = identify_symbol(symbols=[symbol])

        ranges = await get_ranges(ohlc_data=ohlc_data, risk=risk, current_price=price_metadata.get('token0Price'),
                                  token_symbol=token_symbol)

        result = {"pool_id": pool_id, "current_price": round(float(price_metadata.get('token0Price')), 6),
                  **volatility_metrics,
                  **liquidity_metrics,
                  **volume_fee_metrics, **apr_metrics, **ranges, "projected_apr": "projected_apr"}

        return result
    except Exception as e:
        print(f"Error analyze_pool: {e}")
        return f"No Data found for this pool: {pool_id}to analyze"


async def fetch_data_async(session, pool_id, data_type, semaphore, chain):
    """
      Async fetch GraphQL data for a pool.

      Args:
          session: aiohttp session
          pool_id: Target pool ID
          data_type: One of 'price', 'vol', 'liq', 'apr'
          semaphore: Concurrency control
          chain: Blockchain identifier

      Returns:
          dict: JSON response from subgraph
      """
    async with semaphore:
        query = CAMELOT_QUERY_MAPPING[data_type].format(pool_id=pool_id)
        VALID_CHAINS = get_valid_chains()
        url = VALID_CHAINS[chain].format(api_key=SUBGRAPH_API_KEY)
        try:
            async with session.post(url, json={"query": query}) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Error fetching {data_type} for {pool_id}: {response.status}")
                    return {}
        except Exception as e:
            print(f"Exception fetching {data_type} for {pool_id}: {e}")
            return {}


async def analyze_pools_async(pool_ids, chains, risk, symbols):
    """
       Main async analysis entry point.

       Args:
           pool_ids (list): List of pool IDs to analyze
           chains (list): Corresponding chain identifiers

       Returns:
           pd.DataFrame: Analysis results for all pools
       """
    semaphore = asyncio.Semaphore(10)

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [analyze_pool(session, pool_id, semaphore, chain, risk, symbol) for pool_id, chain, symbol in
                 zip(pool_ids, chains, symbols)]
        results = await asyncio.gather(*tasks)

    return pd.DataFrame([r for r in results if r])


# ----------------------------- Utility Functions -----------------------------
def parse_json_string(input_string: str) -> dict:
    """
      Clean and parse JSON from potentially malformed strings.

      Args:
          input_string (str): String containing JSON data

      Returns:
          dict: Parsed JSON data

      Raises:
          ValueError: If invalid JSON after cleaning
      """
    # Remove backticks and words 'json' and 'python'
    cleaned_string = input_string.replace('`', '').replace('json', '').replace('python', '').strip()

    # Remove Python-style comments (lines that start with #)
    cleaned_string = re.sub(r"#.*", "", cleaned_string)

    # Replace tuple-like values with JSON-compatible lists
    cleaned_string = re.sub(r"\(\s*([\d\.\-e]+)\s*,\s*([\d\.\-e]+)\s*\)", r"[\1, \2]", cleaned_string)

    try:
        return json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")


def calculate_indicators(df):
    # Calculate Bollinger Bands (20-day, 2σ)
    df.ta.bbands(length=20, std=2, append=True)

    # Calculate ADX (14-day)
    df.ta.adx(length=14, append=True)

    # Calculate RSI (14-day)
    df.ta.rsi(length=14, append=True)

    # Clean column names
    df.columns = [col.lower() for col in df.columns]

    # Drop rows with NaN values (these are the initial rows without enough data)
    df = df.dropna().reset_index(drop=True)

    return df


def calculate_lp_ranges(df, risk_profile="medium"):
    params = RISK_PARAMS[risk_profile]
    latest = df.iloc[-1]  # Latest data point
    # latest = latest.apply(pd.to_numeric, errors='coerce')  # Convert to float

    # Base Bollinger Bands
    bb_upper = latest['bbu_20_2.0']
    bb_lower = latest['bbl_20_2.0']
    price = latest['close']

    # Volatility-adjusted buffer
    bb_width = bb_upper - bb_lower
    buffer = bb_width * params["volatility_multiplier"]

    # RSI directional bias
    if latest['rsi_14'] > (70 - params["rsi_buffer"]):
        rsi_bias = -0.02  # Overbought: lean lower
    elif latest['rsi_14'] < (30 + params["rsi_buffer"]):
        rsi_bias = 0.02  # Oversold: lean higher
    else:
        rsi_bias = 0

    # ADX-based regime adjustment
    if latest['adx_14'] > params["adx_trend_threshold"]:
        # Trending market: asymmetric range
        trend_strength = latest['adx_14'] / 100
        upper = price * (1 + (0.15 * trend_strength) + rsi_bias)
        lower = price * (1 - (0.10 * trend_strength) + rsi_bias)
    else:
        # Range-bound market: symmetric range
        upper = price + buffer + (price * rsi_bias)
        lower = price - buffer + (price * rsi_bias)

    # Use more decimal places since values are very small
    return round(lower, 6), round(upper, 6)


def identify_symbol(symbols):
    """
    Identifies the primary token category from a list of symbols.

    Args:
        symbols (list): List of token symbols.

    Returns:
        str: "BTC", "ETH", "SOL", or "STABLE" based on the token classification.
    """
    BTC_TOKENS = {"WBTC", "cbBTC", "tBTC", "LBTC", "BTC"}
    ETH_TOKENS = {"ETH", "weETH", "wstETH", "ezETH", "WETH", "SuperOETHb", "msETH", "cbETH"}
    SOL_TOKENS = {"SOL"}

    for symbol in symbols:
        if symbol in BTC_TOKENS:
            return "BTC"

    for symbol in symbols:
        if symbol in SOL_TOKENS:
            return "SOL"

    for symbol in symbols:
        if symbol in ETH_TOKENS:
            return "ETH"

    return "OTHER"


async def fetch_allora_predictions_by_topic(client, token: str):
    """
    Fetches the 8-hour price prediction and volatility prediction from Allora using topic IDs.

    Args:
        token (str): The token symbol (BTC, ETH, SOL).

    Returns:
        dict: A dictionary containing the price and volatility predictions.
    """
    try:
        if token not in ALLORA_TOPIC_ID_MAP:
            raise ValueError(f"Allora does not support predictions for {token}")

        # Fetch price prediction using topic ID
        price_result = await client.get_inference_by_topic_id(ALLORA_TOPIC_ID_MAP[token]["P"])
        price_prediction = price_result.inference_data.network_inference_normalized

        # Fetch volatility prediction using topic ID
        volatility_result = await client.get_inference_by_topic_id(ALLORA_TOPIC_ID_MAP[token]["V"])
        volatility_prediction = volatility_result.inference_data.network_inference_normalized

        return {
            "price_prediction": price_prediction,
            "volatility_prediction": volatility_prediction
        }
    except Exception as e:
        print(f"ERROR: At Allora API {e}")
        return {}


def adjust_ranges_with_predictions(lower, upper, current_price, allora_predictions):
    """
    Adjusts LP ranges based on Allora's price and volatility predictions.

    Args:
        lower (float): Original lower range.
        upper (float): Original upper range.
        current_price (float): Current price of the asset.
        allora_predictions (dict): Dictionary with 'price_prediction' and 'volatility_prediction'.

    Returns:
        tuple: Adjusted (lower, upper) range values.
    """
    price_pred = float(allora_predictions.get("price_prediction", current_price))  # Default to current price if missing
    vol_pred = float(allora_predictions.get("volatility_prediction", 0))  # Default to 0 if missing

    # # Adjust based on price prediction
    # price_change_pct = ((price_pred - current_price) / current_price) * 100
    # if price_change_pct > 0:
    #     upper *= 1 + (price_change_pct / 100)  # Expand upper range
    # else:
    #     lower *= 1 + (price_change_pct / 100)  # Expand lower range

    # Adjust based on volatility prediction
    VOLATILITY_THRESHOLD = 0.005  # Example: Anything above 0.5% is considered high volatility
    if vol_pred > VOLATILITY_THRESHOLD:
        volatility_adjustment = min(vol_pred * 10, 0.10)  # Scale adjustment (capped at 10%)
        lower *= 1 - volatility_adjustment
        upper *= 1 + volatility_adjustment

    return lower, upper


async def get_ranges(ohlc_data, risk, current_price, token_symbol):
    """
    Get LP ranges and adjust based on Allora predictions.
    """
    data = ohlc_data['data']['poolDayDatas']
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df = df.sort_values('date').reset_index(drop=True)
    df[['high', 'low', 'close', 'volumeUSD']] = df[['high', 'low', 'close', 'volumeUSD']].astype(float)

    df = calculate_indicators(df)
    current_price = float(current_price)
    lower, upper = calculate_lp_ranges(df, risk)

    if not token_symbol == "OTHER":
        client = AlloraAPIClient(chain_slug=ChainSlug.TESTNET)
        allora_predictions = await fetch_allora_predictions_by_topic(client, token_symbol)
        if allora_predictions:
            lower, upper = adjust_ranges_with_predictions(lower, upper, current_price, allora_predictions)

    lower_pct = ((lower - current_price) / current_price) * 100
    upper_pct = ((upper - current_price) / current_price) * 100

    return {
        "pool_range": (lower, upper),
        "upper_limit_pct": f"{upper_pct:.3f}%",
        "lower_limit_pct": f"{lower_pct:.3f}%",
        "fee_rank": "🟢 Low" if risk == "low" else "🟡 Medium" if risk == "medium" else "🔴 High"
    }
