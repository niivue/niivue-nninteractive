import { useRef, useCallback, useMemo } from "react";
import {
  DRAG_MODE,
  type DragReleaseParams,
  MULTIPLANAR_TYPE,
  Niivue,
  type NVConfigOptions,
  SHOW_RENDER,
} from "@niivue/niivue";
import { NiivueNNInteractive } from "../niivue-nninteractive";

const API_URL = import.meta.env.VITE_API_URL;

interface UseNiivueInstanceProps {
  nvopts: Partial<NVConfigOptions>;
  onCanvasReady?: (canvas: HTMLCanvasElement, niivue: Niivue) => void;
  onImageLoaded?: (niivue: Niivue, nnInteractive: NiivueNNInteractive) => void;
  onUserIdReceived?: (userId: string) => void;
  onSegmentationComplete?: (result: ArrayBuffer) => void;
  onError?: (error: Error) => void;
}

export const useNiivueInstance = ({
  nvopts,
  onCanvasReady,
  onImageLoaded,
  onUserIdReceived,
  onSegmentationComplete,
  onError,
}: UseNiivueInstanceProps) => {
  const nvRef = useRef<Niivue | null>(null);
  const nnInteractiveRef = useRef<NiivueNNInteractive | null>(null);
  const isInitializedRef = useRef(false);

  // Memoize stable callback functions to prevent dependency changes
  const stableCallbacks = useMemo(
    () => ({
      onCanvasReady,
      onImageLoaded,
      onUserIdReceived,
      onSegmentationComplete,
      onError,
    }),
    [
      onCanvasReady,
      onImageLoaded,
      onUserIdReceived,
      onSegmentationComplete,
      onError,
    ],
  );

  // Initialize Niivue instance once
  if (!nvRef.current) {
    nvRef.current = new Niivue({
      ...nvopts,
      multiplanarShowRender: SHOW_RENDER.ALWAYS,
      multiplanarLayout: MULTIPLANAR_TYPE.GRID,
      dragMode: DRAG_MODE.roiSelection,
    });

    // Setup drag release handler
    nvRef.current.onDragRelease = (dragData: DragReleaseParams) => {
      console.log(dragData);
    };
  }

  const canvasRef = useCallback(
    (node: HTMLCanvasElement | null) => {
      if (node && !isInitializedRef.current) {
        isInitializedRef.current = true;

        const setupCanvas = async (canvas: HTMLCanvasElement) => {
          const niivue = nvRef.current;
          if (!niivue) return;

          niivue.attachToCanvas(canvas);
          niivue.loadVolumes([{ url: "./FLAIR.nii.gz" }]);

          // Create NiivueNNInteractive instance
          const nnInt = new NiivueNNInteractive({
            niivue,
            apiUrl: `${API_URL}/segment`,
            onUserIdReceived: stableCallbacks.onUserIdReceived,
            onSegmentationComplete: stableCallbacks.onSegmentationComplete,
            onError: stableCallbacks.onError,
          });
          nnInteractiveRef.current = nnInt;

          // Set up image loaded callback
          niivue.onImageLoaded = async () => {
            if (niivue.volumes.length > 1) {
              return;
            }

            // Clean up for new image
            niivue.closeDrawing();
            while (niivue.volumes.length > 1) {
              niivue.removeVolume(niivue.volumes[1]);
            }

            // Reset session and prepare for segmentation
            nnInt.resetSession();
            await nnInt.prepareForSegmentation();

            stableCallbacks.onImageLoaded?.(niivue, nnInt);
          };

          stableCallbacks.onCanvasReady?.(canvas, niivue);
        };
        setupCanvas(node);
      }
    },
    [stableCallbacks],
  );

  return {
    niivue: nvRef.current,
    nnInteractive: nnInteractiveRef.current,
    canvasRef,
  };
};

