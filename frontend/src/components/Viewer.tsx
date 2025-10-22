import React from "react";
import type { NVConfigOptions } from "@niivue/niivue";

interface ViewerProps {
  imageUrl: string;
  nvopts: Partial<NVConfigOptions>;
  canvasRef: (node: HTMLCanvasElement | null) => void;
}

export const Viewer = React.memo<ViewerProps>(({ canvasRef }) => (
  <canvas
    ref={canvasRef}
    height={480}
    width={640}
    style={{ outline: "none", border: "none" }}
  />
));