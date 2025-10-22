import { useState, useCallback } from "react";
import { NiivueNNInteractive } from "../niivue-nninteractive";

interface SegmentationState {
  isLoading: boolean;
  drawingEnabled: boolean;
  negativeMode: boolean;
  userId: string | null;
  segmentationResult: ArrayBuffer | null;
}

export const useSegmentationState = () => {
  const [state, setState] = useState<SegmentationState>({
    isLoading: false,
    drawingEnabled: false,
    negativeMode: true,
    userId: null,
    segmentationResult: null,
  });

  const setIsLoading = useCallback((isLoading: boolean) => {
    setState(prev => ({ ...prev, isLoading }));
  }, []);

  const setDrawingEnabled = useCallback((drawingEnabled: boolean) => {
    setState(prev => ({ ...prev, drawingEnabled }));
  }, []);

  const setNegativeMode = useCallback((negativeMode: boolean) => {
    setState(prev => ({ ...prev, negativeMode }));
  }, []);

  const setUserId = useCallback((userId: string | null) => {
    setState(prev => ({ ...prev, userId }));
  }, []);

  const setSegmentationResult = useCallback((segmentationResult: ArrayBuffer | null) => {
    setState(prev => ({ ...prev, segmentationResult }));
  }, []);

  const resetForNewImage = useCallback(() => {
    setState(prev => ({
      ...prev,
      userId: null,
      segmentationResult: null,
      negativeMode: true,
      drawingEnabled: true,
    }));
  }, []);

  const handleSegment = useCallback(async (nnInteractive: NiivueNNInteractive | null) => {
    if (state.isLoading || !nnInteractive) return;

    setIsLoading(true);
    try {
      await nnInteractive.segmentAndLoad();
    } catch {
      // Error handling is done in the NiivueNNInteractive class
    } finally {
      setIsLoading(false);
    }
  }, [state.isLoading, setIsLoading]);

  const handleToggleDrawing = useCallback((nnInteractive: NiivueNNInteractive | null) => {
    if (!nnInteractive) return;

    const newState = !state.drawingEnabled;
    setDrawingEnabled(newState);
    nnInteractive.setDrawingEnabled(newState);

    if (newState) {
      nnInteractive.setPenValue(state.negativeMode);
    }
  }, [state.drawingEnabled, state.negativeMode, setDrawingEnabled]);

  const handleToggleMode = useCallback((nnInteractive: NiivueNNInteractive | null) => {
    const newMode = !state.negativeMode;
    setNegativeMode(newMode);
    if (nnInteractive && state.drawingEnabled) {
      nnInteractive.setPenValue(newMode);
    }
  }, [state.negativeMode, state.drawingEnabled, setNegativeMode]);

  const handleClearDrawing = useCallback((nnInteractive: NiivueNNInteractive | null) => {
    if (!nnInteractive) return;
    nnInteractive.clearDrawing();
  }, []);

  const handleDownloadSegmentation = useCallback(() => {
    if (!state.segmentationResult) return;

    const blob = new Blob([state.segmentationResult], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "segmentation.nii.gz";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [state.segmentationResult]);

  return {
    ...state,
    resetForNewImage,
    setUserId,
    setSegmentationResult,
    handleSegment,
    handleToggleDrawing,
    handleToggleMode,
    handleClearDrawing,
    handleDownloadSegmentation,
  };
};