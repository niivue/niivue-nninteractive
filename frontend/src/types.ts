import type { Niivue } from "@niivue/niivue";

export interface ScribbleCoordinate {
  x: number;
  y: number;
  z: number;
  is_positive: boolean;
}

export interface SegmentationOptions {
  apiUrl: string;
  userId?: string;
  onUserIdReceived?: (userId: string) => void;
  onSegmentationComplete?: (result: ArrayBuffer) => void;
  onError?: (error: Error) => void;
}

export interface NiivueNNInteractiveConfig {
  niivue: Niivue;
  apiUrl?: string;
  onUserIdReceived?: (userId: string) => void;
  onSegmentationComplete?: (result: ArrayBuffer) => void;
  onError?: (error: Error) => void;
}

export interface SegmentationResponse {
  arrayBuffer: ArrayBuffer;
  userId?: string;
}