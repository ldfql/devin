import { create } from 'zustand';

interface WebSocketStore {
  connected: boolean;
  lastMessage: any;
  connect: () => void;
  disconnect: () => void;
  sendMessage: (message: any) => void;
}

let ws: WebSocket | null = null;
let reconnectTimeout: NodeJS.Timeout | null = null;
const RECONNECT_DELAY = 5000; // 5 seconds
const MAX_RETRIES = 5;
let retryCount = 0;

export const useWebSocket = create<WebSocketStore>((set) => ({
  connected: false,
  lastMessage: null,
  connect: () => {
    if (ws?.readyState === WebSocket.OPEN) return;

    const wsUrl = `${import.meta.env.VITE_API_URL.replace('http', 'ws')}/ws`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      set({ connected: true });
      console.log('WebSocket connected');
      retryCount = 0; // Reset retry count on successful connection
    };

    ws.onclose = () => {
      set({ connected: false });
      console.log('WebSocket disconnected');

      // Clear existing timeout if any
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }

      // Attempt to reconnect if under max retries
      if (retryCount < MAX_RETRIES) {
        retryCount++;
        reconnectTimeout = setTimeout(() => {
          console.log(`Reconnecting... Attempt ${retryCount}/${MAX_RETRIES}`);
          useWebSocket.getState().connect();
        }, RECONNECT_DELAY);
      } else {
        console.error('Max reconnection attempts reached');
      }
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        set({ lastMessage: message });
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  },
  disconnect: () => {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }
    if (ws) {
      ws.close();
      ws = null;
    }
    retryCount = 0;
    set({ connected: false });
  },
  sendMessage: (message: any) => {
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  },
}));
