#!/usr/bin/env python3
from app.services.engine import engine
import time

print("Testing engine...")
# Reset engine state for consistency? We'll just use as is.
for i in range(5):
    reading = engine.generate_reading()
    print(f"Tick {i}: HR={reading['vitals']['heart_rate']}, Pos=({reading['vitals']['position']['x']:.2f}, {reading['vitals']['position']['z']:.2f})")
print("Test completed.")