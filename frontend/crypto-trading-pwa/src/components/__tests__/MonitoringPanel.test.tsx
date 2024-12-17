import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MonitoringPanel } from '../MonitoringPanel';

describe('MonitoringPanel', () => {
  it('renders market data', () => {
    render(<MonitoringPanel />);
    expect(screen.getByText(/Market Data/i)).toBeDefined();
  });

  it('displays trading signals', () => {
    const mockSignal = {
      symbol: 'BTCUSDT',
      direction: 'BUY',
      confidence: 0.85,
      entry: 45000,
      stopLoss: 44000,
      target: 48000
    };

    render(<MonitoringPanel initialSignal={mockSignal} />);
    expect(screen.getByText(/BTCUSDT/i)).toBeDefined();
    expect(screen.getByText(/BUY/i)).toBeDefined();
    expect(screen.getByText(/85%/i)).toBeDefined();
  });

  it('handles position sizing calculations', () => {
    const mockBalance = 10000;
    render(<MonitoringPanel accountBalance={mockBalance} />);
    const positionSize = screen.getByTestId('position-size');
    expect(positionSize.textContent).toMatch(/Position Size:/);
  });

  it('updates real-time data', async () => {
    const mockWebSocket = {
      send: vi.fn(),
      addEventListener: vi.fn()
    };

    vi.spyOn(window, 'WebSocket').mockImplementation(() => mockWebSocket as any);

    render(<MonitoringPanel />);
    const priceElement = screen.getByTestId('btc-price');
    expect(priceElement).toBeDefined();
  });
});
