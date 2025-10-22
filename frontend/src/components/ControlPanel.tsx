import React from "react";
import { Pencil, Send, Download, Eraser, Plus, Minus } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Niivue } from "@niivue/niivue";
import type { NiivueNNInteractive } from "../niivue-nninteractive";

interface ControlPanelProps {
  niivue: Niivue | null;
  nnInteractive: NiivueNNInteractive | null;
  isLoading: boolean;
  drawingEnabled: boolean;
  negativeMode: boolean;
  segmentationResult: ArrayBuffer | null;
  onToggleDrawing: (nnInteractive: NiivueNNInteractive | null) => void;
  onToggleMode: (nnInteractive: NiivueNNInteractive | null) => void;
  onClearDrawing: (nnInteractive: NiivueNNInteractive | null) => void;
  onSegment: (nnInteractive: NiivueNNInteractive | null) => Promise<void>;
  onDownload: () => void;
}

export const ControlPanel = React.memo<ControlPanelProps>(({
  niivue,
  nnInteractive,
  isLoading,
  drawingEnabled,
  negativeMode,
  segmentationResult,
  onToggleDrawing,
  onToggleMode,
  onClearDrawing,
  onSegment,
  onDownload,
}) => {
  const isSegmentDisabled = isLoading || !niivue || !nnInteractive;

  return (
    <div className="bg-slate-900 border-t border-slate-700 p-4">
      <div className="flex flex-wrap justify-center items-center gap-3">
        {/* Drawing Control Group */}
        <div className="flex items-center gap-2 bg-slate-800 rounded-lg p-2 border border-slate-600">
          <Button
            onClick={() => onToggleDrawing(nnInteractive)}
            variant={drawingEnabled ? "default" : "outline"}
            size="sm"
            className={drawingEnabled ? 
              "bg-slate-600 hover:bg-slate-500 text-slate-100 border-slate-600" : 
              "border-slate-500 hover:bg-slate-700 text-slate-200"
            }
          >
            <Pencil size={16} />
            <span className="ml-2 font-medium">
              {drawingEnabled ? "Drawing On" : "Drawing Off"}
            </span>
          </Button>
          
          <Button
            onClick={() => onToggleMode(nnInteractive)}
            variant={negativeMode ? "default" : "outline"}
            size="sm"
            disabled={!drawingEnabled}
            className={
              !drawingEnabled ? "opacity-50 border-slate-600 text-slate-400" :
              negativeMode ? 
                "bg-slate-600 hover:bg-slate-500 text-slate-100 border-slate-600" : 
                "border-slate-500 hover:bg-slate-700 text-slate-200"
            }
          >
            {negativeMode ? <Plus size={16} /> : <Minus size={16} />}
            <span className="ml-2 font-medium">
              {negativeMode ? "Include" : "Exclude"}
            </span>
          </Button>
          
          <Button
            onClick={() => onClearDrawing(nnInteractive)}
            variant="outline"
            size="sm"
            disabled={!drawingEnabled}
            className="border-slate-500 hover:bg-slate-700 text-slate-200 disabled:opacity-50 disabled:text-slate-400"
          >
            <Eraser size={16} />
            <span className="ml-2">Clear</span>
          </Button>
        </div>
        
        {/* Action Buttons */}
        <Button
          onClick={() => onSegment(nnInteractive)}
          disabled={isSegmentDisabled}
          size="lg"
          className={
            isSegmentDisabled ? 
              "bg-slate-700 text-slate-400 cursor-not-allowed border-slate-600" : 
              "bg-slate-600 hover:bg-slate-500 text-slate-100 font-semibold border-slate-600"
          }
        >
          <Send size={18} />
          <span className="ml-2">
            {isLoading ? "Processing..." : "Run Segmentation"}
          </span>
        </Button>
        
        {segmentationResult && (
          <Button
            onClick={onDownload}
            variant="outline"
            size="lg"
            className="border-slate-500 hover:bg-slate-700 text-slate-200 font-medium"
          >
            <Download size={18} />
            <span className="ml-2">Download Result</span>
          </Button>
        )}
      </div>
    </div>
  );
});