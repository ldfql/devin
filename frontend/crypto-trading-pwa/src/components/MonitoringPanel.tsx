import React from 'react';
import { useWebSocket } from '../services/websocket';
import { Card, CardContent } from './ui/card';
import { Alert, AlertDescription } from './ui/alert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface CryptoData {
  symbol: string;
  price: number;
  change24h: number;
  timestamp: number;
}

interface TradingSignal {
  symbol: string;
  direction: 'buy' | 'sell';
  confidence: number;
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  timestamp: number;
}

export const MonitoringPanel: React.FC = () => {
  const { lastMessage } = useWebSocket();
  const [priceHistory, setPriceHistory] = React.useState<CryptoData[]>([]);
  const [signals, setSignals] = React.useState<TradingSignal[]>([]);

  React.useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'price_update') {
        setPriceHistory(prev => [...prev.slice(-30), lastMessage.data]);
      } else if (lastMessage.type === 'trading_signal') {
        setSignals(prev => [...prev.slice(-5), lastMessage.data]);
      }
    }
  }, [lastMessage]);

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="p-6">
          <h3 className="text-lg font-medium mb-4">Price Chart</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={priceHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(timestamp) => new Date(timestamp).toLocaleTimeString()}
                />
                <YAxis />
                <Tooltip
                  labelFormatter={(timestamp) => new Date(timestamp).toLocaleString()}
                  formatter={(value: number) => [`$${value.toFixed(2)}`, 'Price']}
                />
                <Line
                  type="monotone"
                  dataKey="price"
                  stroke="#2563eb"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="space-y-4">
        <h3 className="text-lg font-medium">Trading Signals</h3>
        {signals.map((signal, index) => (
          <Alert key={index} variant={signal.direction === 'buy' ? 'default' : 'destructive'}>
            <AlertDescription>
              <div className="flex justify-between items-center">
                <div>
                  <span className="font-medium">{signal.symbol}</span>
                  <span className="ml-2 text-sm">
                    {signal.direction.toUpperCase()} @ ${signal.entryPrice.toFixed(2)}
                  </span>
                </div>
                <div className="text-sm">
                  <div>SL: ${signal.stopLoss.toFixed(2)}</div>
                  <div>TP: ${signal.takeProfit.toFixed(2)}</div>
                </div>
                <div className="text-sm">
                  Confidence: {(signal.confidence * 100).toFixed(1)}%
                </div>
              </div>
            </AlertDescription>
          </Alert>
        ))}
      </div>
    </div>
  );
};
