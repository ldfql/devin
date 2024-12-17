import '@testing-library/jest-dom';
import { afterEach, beforeEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

// Mock environment variables
const env = {
  VITE_API_URL: 'http://localhost:8000'
};

// Setup fetch mock
const fetchMock = vi.fn();

beforeEach(() => {
  // Setup environment variables
  vi.stubGlobal('import.meta', { env });

  // Reset the mock but keep the mock implementation
  fetchMock.mockReset();
  // Mock global fetch
  global.fetch = fetchMock;
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

// Export the mocked fetch for use in tests
export const mockFetch = fetchMock;
