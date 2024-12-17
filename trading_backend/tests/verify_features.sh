#!/bin/bash

echo "Testing Chinese Platform Integration..."
curl -X GET "http://localhost:8000/api/chinese-platforms/sentiment/BTCUSDT" -H "accept: application/json"
echo -e "\n---\n"

echo "Testing Screenshot Analysis..."
curl -X POST "http://localhost:8000/api/screenshot/analyze" \
  -H "accept: application/json" \
  -F "file=@data/test_screenshot_cn.txt"
echo -e "\n---\n"

echo "Testing Market Analysis..."
curl -X GET "http://localhost:8000/api/market-analysis/analyze/BTCUSDT" -H "accept: application/json"
echo -e "\n---\n"

echo "Testing Notification Settings..."
curl -X POST "http://localhost:8000/api/notification/email" \
  -H "Content-Type: application/json" \
  -d '{
    "template_type": "trading_signal",
    "recipient": "test@example.com",
    "params": {
      "pair": "BTCUSDT",
      "signal_type": "BUY",
      "entry_price": "45000",
      "stop_loss": "44000",
      "take_profit": "48000",
      "confidence": "85"
    }
  }'
