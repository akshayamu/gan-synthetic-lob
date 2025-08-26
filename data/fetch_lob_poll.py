import ccxt, time, numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--symbol", type=str, default="BTC/USDT")
parser.add_argument("--minutes", type=int, default=1)
parser.add_argument("--levels", type=int, default=10)
parser.add_argument("--out", type=str, default="data/lob.npy")
args = parser.parse_args()

binance = ccxt.binance()

snapshots = []
interval = 5  # seconds
steps = args.minutes * 60 // interval

print(f"Recording {args.minutes} min of {args.symbol} at 5s intervals...")

for i in range(steps):
    lob = binance.fetch_order_book(args.symbol, limit=args.levels)
    bids = np.array(lob['bids'][:args.levels])
    asks = np.array(lob['asks'][:args.levels])
    snapshots.append([bids, asks])
    print(f"✅ snapshot {i+1}/{steps}")
    time.sleep(interval)

np.save(args.out, np.array(snapshots, dtype=object))
print(f"Done! Saved {len(snapshots)} snapshots to {args.out}")


