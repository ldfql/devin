import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ScreenshotUpload } from '../ScreenshotUpload';
import { describe, it, expect, beforeEach } from 'vitest';
import { mockFetch } from '../../test/setup';

describe('ScreenshotUpload', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it('renders upload button', () => {
    render(<ScreenshotUpload />);
    expect(screen.getByText(/Drag & drop screenshot here/i)).toBeInTheDocument();
  });

  it('handles file upload', async () => {
    mockFetch.mockImplementationOnce(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ text: 'Analyzed text', confidence: 0.95 })
      } as Response)
    );

    const file = new File(['test image'], 'test.png', { type: 'image/png' });
    render(<ScreenshotUpload />);

    const input = screen.getByTestId('file-input');
    fireEvent.change(input, { target: { files: [file] } });

    // Wait for uploading state
    await waitFor(() => {
      expect(screen.getByText(/Analyzing screenshot/i)).toBeInTheDocument();
    });

    // Wait for uploading state to finish and results to appear
    await waitFor(() => {
      expect(screen.queryByText(/Analyzing screenshot/i)).not.toBeInTheDocument();
      expect(screen.getByText(/Analysis Results/)).toBeInTheDocument();
      expect(screen.getByText(/Analyzed text/)).toBeInTheDocument();
      expect(screen.getByText(/95%/)).toBeInTheDocument();
    });
  });

  it('displays error on failed upload', async () => {
    mockFetch.mockImplementationOnce(() =>
      Promise.resolve({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      } as Response)
    );

    const file = new File(['test image'], 'test.png', { type: 'image/png' });
    render(<ScreenshotUpload />);

    const input = screen.getByTestId('file-input');
    fireEvent.change(input, { target: { files: [file] } });

    // Wait for error message
    await waitFor(() => {
      expect(screen.getByText(/Failed to analyze screenshot/i)).toBeInTheDocument();
    });
  });
});
