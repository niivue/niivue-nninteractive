import type { Niivue } from "@niivue/niivue";
import type {
  ScribbleCoordinate,
  SegmentationOptions,
  NiivueNNInteractiveConfig,
  SegmentationResponse,
} from "./types";

export * from "./types";

export class NiivueNNInteractive {
  private niivue: Niivue;
  private apiUrl: string;
  private userId: string | null = null;
  private onUserIdReceived?: (userId: string) => void;
  private onSegmentationComplete?: (result: ArrayBuffer) => void;
  private onError?: (error: Error) => void;

  constructor(config: NiivueNNInteractiveConfig) {
    this.niivue = config.niivue;
    this.apiUrl = config.apiUrl || "http://localhost:8000/segment";
    this.onUserIdReceived = config.onUserIdReceived;
    this.onSegmentationComplete = config.onSegmentationComplete;
    this.onError = config.onError;
  }

  /**
   * Extract scribble coordinates from the current Niivue drawing
   */
  extractScribbles(): ScribbleCoordinate[] {
    const scribbles: ScribbleCoordinate[] = [];
    const drawBitmap = this.niivue.drawBitmap;

    if (!drawBitmap || drawBitmap.length === 0) return [];

    // Get volume dimensions
    const volume = this.niivue.volumes[0];
    if (!volume || !volume.dims) return [];

    const dims = volume.dims.slice(1, 4); // [x, y, z] dimensions

    // Iterate through the 1D array to find non-zero values
    for (let i = 0; i < drawBitmap.length; i++) {
      if (drawBitmap[i] > 0) {
        // Convert 1D index to 3D coordinates
        const z = Math.floor(i / (dims[0] * dims[1]));
        const y = Math.floor((i % (dims[0] * dims[1])) / dims[0]);
        const x = i % dims[0];

        scribbles.push({
          x,
          y,
          z,
          is_positive: drawBitmap[i] === 2, // 2 for positive, 1 for negative
        });
      }
    }

    return scribbles;
  }

  /**
   * Perform segmentation using the nnInteractive API
   */
  async performSegmentation(options?: Partial<SegmentationOptions>): Promise<SegmentationResponse> {
    const apiUrl = options?.apiUrl || this.apiUrl;
    const userId = options?.userId || this.userId;

    try {
      const scribbles = this.extractScribbles();

      if (scribbles.length === 0) {
        throw new Error("No scribbles found. Please draw on the image first.");
      }

      // Log scribble coordinates
      console.log(`Total scribbles: ${scribbles.length}`);
      console.log("Full scribble array:", scribbles);
      
      // Separate and log positive/negative scribbles
      const positiveScribbles = scribbles.filter(s => s.is_positive);
      const negativeScribbles = scribbles.filter(s => !s.is_positive);
      console.log(`Positive scribbles (${positiveScribbles.length}):`, positiveScribbles);
      console.log(`Negative scribbles (${negativeScribbles.length}):`, negativeScribbles);
      
      // Log JSON string that will be sent
      const scribblesJson = JSON.stringify(scribbles);
      console.log("JSON string being sent to API:", scribblesJson);
      console.log("================================================");

      const formData = new FormData();

      // If no userId, this is the first request and we need to send the image
      if (!userId) {
        // Get the current volume file
        const volume = this.niivue.volumes[0];
        if (!volume) {
          throw new Error("No volume loaded");
        }

        // Convert the current volume to a blob
        const uint8Array = await volume.saveToUint8Array("image.nii.gz");
        const blob = new Blob([new Uint8Array(uint8Array)], { type: "application/octet-stream" });
        formData.append("image", blob, "image.nii.gz");
      } else {
        formData.append("user_id", userId);
      }

      formData.append("scribbles", JSON.stringify(scribbles));

      const response = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Get user ID from response headers if this is the first request
      let newUserId: string | undefined;
      if (!userId) {
        newUserId = response.headers.get("X-User-ID") || undefined;
        if (newUserId) {
          this.userId = newUserId;
          this.onUserIdReceived?.(newUserId);
          options?.onUserIdReceived?.(newUserId);
        }
      }

      // Get the segmentation result
      const segmentationBlob = await response.blob();
      const arrayBuffer = await segmentationBlob.arrayBuffer();

      this.onSegmentationComplete?.(arrayBuffer);
      options?.onSegmentationComplete?.(arrayBuffer);

      return { arrayBuffer, userId: newUserId };
    } catch (error) {
      const err = error instanceof Error ? error : new Error("Unknown error");
      this.onError?.(err);
      options?.onError?.(err);
      throw err;
    }
  }

  /**
   * Load segmentation result as an overlay in Niivue
   */
  async loadSegmentationOverlay(
    arrayBuffer: ArrayBuffer,
    options?: {
      colormap?: string;
      opacity?: number;
    }
  ): Promise<void> {
    const blob = new Blob([arrayBuffer], { type: "application/octet-stream" });
    const segmentationUrl = URL.createObjectURL(blob);

    await this.niivue.addVolumeFromUrl({
      url: segmentationUrl,
      colormap: options?.colormap || "red",
      opacity: options?.opacity || 0.5,
    });

    // Clean up the object URL after loading
    URL.revokeObjectURL(segmentationUrl);
  }

  clearDrawing(): void {
      this.niivue.closeDrawing();
      this.niivue.createEmptyDrawing();
  }

  setPenValue(isPositive: boolean): void {
    this.niivue.setPenValue(isPositive ? 2 : 1);
  }

  setDrawingEnabled(enabled: boolean): void {
    this.niivue.setDrawingEnabled(enabled);
  }

  getUserId(): string | null {
    return this.userId;
  }

  setUserId(userId: string | null): void {
    this.userId = userId;
  }

  resetSession(): void {
    this.userId = null;
    this.clearDrawing();
  }

  /**
   * Prepare Niivue for interactive segmentation
   * This sets up the drawing overlay and enables drawing
   */
  async prepareForSegmentation(): Promise<void> {
    // Clone the loaded image to create an overlay that stores the segmentation
    const overlay = this.niivue.cloneVolume(0);
    overlay.img?.fill(0); // fill with zeros since it will hold binary segmentation later
    overlay.opacity = 0.5;
    overlay.colormap = "red";

    // Add the overlay to niivue
    this.niivue.addVolume(overlay);

    // Create empty drawing and enable drawing
    this.niivue.createEmptyDrawing();
    this.niivue.setDrawingEnabled(true);

    // Set initial drawing pen value to positive (2) for "Include" mode
    this.niivue.setPenValue(2);
    this.niivue.drawOpacity = 1.0; // Ensure drawing is visible
  }

  /**
   * Convenience method to perform segmentation and load the resultI
   */
  async segmentAndLoad(options?: Partial<SegmentationOptions> & {
    colormap?: string;
    opacity?: number;
  }): Promise<void> {
    const result = await this.performSegmentation(options);
    await this.loadSegmentationOverlay(result.arrayBuffer, {
      colormap: options?.colormap,
      opacity: options?.opacity,
    });
  }
}