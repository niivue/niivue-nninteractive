import React from "react";
import { SLICE_TYPE } from "@niivue/niivue";
import { useNiivueInstance, useSegmentationState, useToolbarState } from "./hooks";
import { LeftSidebar, StatusDisplay, Viewer } from "./components";

const App = React.memo(() => {
  const segmentationState = useSegmentationState();
  const toolbarState = useToolbarState();

  const { niivue, nnInteractive, canvasRef } = useNiivueInstance({
    nvopts: { sliceType: SLICE_TYPE.AXIAL, crosshairWidth: 0 },
    onUserIdReceived: segmentationState.setUserId,
    onSegmentationComplete: segmentationState.setSegmentationResult,
    onImageLoaded: () => {
      segmentationState.resetForNewImage();
    },
    onError: (error) => {
      console.error("Error performing segmentation:", error);
      alert(`Error: ${error.message}`);
    },
  });

  return (
    <div className="dark">
      <div className="h-screen w-screen bg-black overflow-hidden">
        <div className="grid grid-cols-[auto_1fr] h-screen w-full">
          <LeftSidebar
            niivue={niivue}
            nnInteractive={nnInteractive}
            isLoading={segmentationState.isLoading}
            drawingEnabled={segmentationState.drawingEnabled}
            negativeMode={segmentationState.negativeMode}
            segmentationResult={segmentationState.segmentationResult}
            onSliceClick={toolbarState.handleSliceClick}
            onCrosshairClick={toolbarState.handleCrosshairClick}
            onToggleDrawing={segmentationState.handleToggleDrawing}
            onToggleMode={segmentationState.handleToggleMode}
            onClearDrawing={segmentationState.handleClearDrawing}
            onSegment={segmentationState.handleSegment}
            onDownload={segmentationState.handleDownloadSegmentation}
          />

          {/* Main Content Area */}
          <div className="flex flex-col h-screen min-w-0">
            <div className="bg-black flex-1 min-h-0 flex items-center justify-center">
              <Viewer
                imageUrl=""
                nvopts={{ sliceType: SLICE_TYPE.AXIAL, crosshairWidth: 0 }}
                canvasRef={canvasRef}
              />
            </div>
            
            {/* Status Display at Bottom */}
            <div className="shrink-0">
              <StatusDisplay userId={segmentationState.userId} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

export default App;