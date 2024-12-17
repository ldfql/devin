import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image as ImageIcon } from 'lucide-react';
import { Card } from './ui/card';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';

interface ScreenshotUploadProps {
  onUploadComplete?: (result: { text: string; confidence: number }) => void;
}

interface AnalysisResult {
  text: string;
  confidence: number;
}

export const ScreenshotUpload: React.FC<ScreenshotUploadProps> = ({ onUploadComplete }) => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploading(true);
    setProgress(0);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${import.meta.env.VITE_API_URL}/screenshot/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to analyze screenshot');
      }

      const analysisResult = await response.json();
      setResult(analysisResult);
      onUploadComplete?.(analysisResult);
      setProgress(100);
    } catch (err) {
      if (!navigator.onLine) {
        setError('You are offline. Please check your internet connection.');
      } else {
        setError(err instanceof Error ? err.message : 'Failed to upload screenshot');
      }
    } finally {
      setUploading(false);
    }
  }, [onUploadComplete]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    maxFiles: 1
  });

  return (
    <Card className="p-6">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-primary bg-primary/10' : 'border-gray-300 hover:border-primary'}`}
      >
        <input
          {...getInputProps()}
          data-testid="file-input"
          aria-label="Upload screenshot"
          style={{ display: 'none' }}
        />
        <div className="flex flex-col items-center gap-4">
          {isDragActive ? (
            <Upload className="w-12 h-12 text-primary animate-bounce" />
          ) : (
            <ImageIcon className="w-12 h-12 text-gray-400" />
          )}
          <div className="space-y-1">
            <p className="text-sm font-medium">
              {isDragActive ? 'Drop the screenshot here' : 'Drag & drop screenshot here'}
            </p>
            <p className="text-xs text-gray-500">
              or click to select file
            </p>
          </div>
        </div>
      </div>

      {uploading && (
        <div className="mt-4 space-y-2">
          <Progress value={progress} className="w-full" />
          <p className="text-sm text-center text-gray-500">
            Analyzing screenshot...
          </p>
        </div>
      )}

      {error && (
        <Alert variant="default" className="mt-4">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {result && (
        <div className="mt-4 space-y-2">
          <h3 className="text-lg font-semibold">Analysis Results</h3>
          <div className="p-4 bg-gray-50 rounded-lg">
            <p className="whitespace-pre-wrap font-mono text-sm">{result.text}</p>
            <p className="text-sm text-gray-500 mt-2">
              Confidence: {Math.round(result.confidence * 100)}%
            </p>
          </div>
        </div>
      )}
    </Card>
  );
};
