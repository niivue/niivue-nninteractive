import React, { useState } from "react";
import { Layers, Crosshair, Pencil, Send, Download, Eraser, Plus, Minus, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Niivue } from "@niivue/niivue";
import type { NiivueNNInteractive } from "../niivue-nninteractive";

interface LeftSidebarProps {
  niivue: Niivue | null;
  nnInteractive: NiivueNNInteractive | null;
  isLoading: boolean;
  drawingEnabled: boolean;
  negativeMode: boolean;
  segmentationResult: ArrayBuffer | null;
  onSliceClick: (niivue: Niivue | null) => void;
  onCrosshairClick: (niivue: Niivue | null) => void;
  onToggleDrawing: (nnInteractive: NiivueNNInteractive | null) => void;
  onToggleMode: (nnInteractive: NiivueNNInteractive | null) => void;
  onClearDrawing: (nnInteractive: NiivueNNInteractive | null) => void;
  onSegment: (nnInteractive: NiivueNNInteractive | null) => Promise<void>;
  onDownload: () => void;
}

export const LeftSidebar = React.memo<LeftSidebarProps>(({
  niivue,
  nnInteractive,
  isLoading,
  drawingEnabled,
  negativeMode,
  segmentationResult,
  onSliceClick,
  onCrosshairClick,
  onToggleDrawing,
  onToggleMode,
  onClearDrawing,
  onSegment,
  onDownload,
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const isSegmentDisabled = isLoading || !niivue || !nnInteractive;

  return (
    <div className={`relative bg-slate-100 dark:bg-slate-900 border-r border-slate-300 dark:border-slate-700 transition-all duration-300 ${
      isCollapsed ? 'w-12' : 'w-80'
    }`}>
      {/* Collapse/Expand Button */}
      <Button
        onClick={() => setIsCollapsed(!isCollapsed)}
        variant="ghost"
        size="icon"
        className="absolute -right-3 top-4 z-10 h-6 w-6 rounded-full bg-slate-200 dark:bg-slate-800 border border-slate-300 dark:border-slate-600 hover:bg-slate-300 dark:hover:bg-slate-700"
        aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        {isCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
      </Button>

      {!isCollapsed && (
        <div className="flex flex-col h-full p-4">
          {/* Top Section - Viewer Tools */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-slate-600 dark:text-slate-300 mb-3 uppercase tracking-wide">
              Viewer Tools
            </h3>
            <div className="space-y-2">
              <Button
                onClick={() => onSliceClick(niivue)}
                variant="outline"
                className="w-full justify-start gap-3 h-10 border-slate-300 dark:border-slate-600 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200"
                title="Toggle slice view"
              >
                <Layers size={18} />
                <span>Slice View</span>
              </Button>
              <Button
                onClick={() => onCrosshairClick(niivue)}
                variant="outline"
                className="w-full justify-start gap-3 h-10 border-slate-300 dark:border-slate-600 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200"
                title="Toggle crosshair"
              >
                <Crosshair size={18} />
                <span>Crosshair</span>
              </Button>
            </div>
          </div>

          {/* Middle Section - Drawing & Annotation Tools */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-slate-600 dark:text-slate-300 mb-3 uppercase tracking-wide">
              Drawing Controls
            </h3>
            <div className="space-y-3">
              <Button
                onClick={() => onToggleDrawing(nnInteractive)}
                variant={drawingEnabled ? "default" : "outline"}
                className={`w-full justify-start gap-3 h-10 ${
                  drawingEnabled
                    ? "bg-slate-600 hover:bg-slate-500 text-slate-100 border-slate-600"
                    : "border-slate-300 dark:border-slate-600 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200"
                }`}
              >
                <Pencil size={18} />
                <span className="font-medium">
                  {drawingEnabled ? "Drawing On" : "Drawing Off"}
                </span>
              </Button>

              <Button
                onClick={() => onToggleMode(nnInteractive)}
                variant={negativeMode ? "default" : "outline"}
                disabled={!drawingEnabled}
                className={`w-full justify-start gap-3 h-10 ${
                  !drawingEnabled
                    ? "opacity-50 border-slate-300 dark:border-slate-600 text-slate-400"
                    : negativeMode
                    ? "bg-slate-600 hover:bg-slate-500 text-slate-100 border-slate-600"
                    : "border-slate-300 dark:border-slate-600 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200"
                }`}
              >
                {negativeMode ? <Plus size={18} /> : <Minus size={18} />}
                <span className="font-medium">
                  {negativeMode ? "Include" : "Exclude"}
                </span>
              </Button>

              <Button
                onClick={() => onClearDrawing(nnInteractive)}
                variant="outline"
                disabled={!drawingEnabled}
                className="w-full justify-start gap-3 h-10 border-slate-300 dark:border-slate-600 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200 disabled:opacity-50 disabled:text-slate-400"
              >
                <Eraser size={18} />
                <span>Clear Drawing</span>
              </Button>
            </div>
          </div>

          {/* Bottom Section - Primary Actions */}
          <div className="mt-auto">
            <div className="space-y-3">
              <Button
                onClick={() => onSegment(nnInteractive)}
                disabled={isSegmentDisabled}
                size="lg"
                className={`w-full h-12 font-semibold text-base ${
                  isSegmentDisabled
                    ? "bg-slate-300 dark:bg-slate-700 text-slate-500 dark:text-slate-400 cursor-not-allowed border-slate-300 dark:border-slate-600"
                    : "bg-blue-600 hover:bg-blue-500 text-white border-blue-600 shadow-lg"
                }`}
              >
                <Send size={20} />
                <span className="ml-2">
                  {isLoading ? "Processing..." : "Run Segmentation"}
                </span>
              </Button>

              {segmentationResult && (
                <Button
                  onClick={onDownload}
                  variant="outline"
                  size="lg"
                  className="w-full h-10 border-slate-300 dark:border-slate-600 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-200 font-medium"
                >
                  <Download size={18} />
                  <span className="ml-2">Download Result</span>
                </Button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Collapsed State - Minimal Icons */}
      {isCollapsed && (
        <div className="flex flex-col items-center pt-12 pb-4 space-y-3">
          <Button
            onClick={() => onSliceClick(niivue)}
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-800"
            title="Toggle slice view"
          >
            <Layers size={16} />
          </Button>
          <Button
            onClick={() => onCrosshairClick(niivue)}
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-800"
            title="Toggle crosshair"
          >
            <Crosshair size={16} />
          </Button>
          
          <div className="h-px w-6 bg-slate-300 dark:bg-slate-600 my-2"></div>
          
          <Button
            onClick={() => onToggleDrawing(nnInteractive)}
            variant="ghost"
            size="icon"
            className={`h-8 w-8 ${
              drawingEnabled
                ? "bg-slate-600 text-white hover:bg-slate-500"
                : "text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-800"
            }`}
            title={drawingEnabled ? "Drawing On" : "Drawing Off"}
          >
            <Pencil size={16} />
          </Button>
          
          <div className="mt-auto">
            <Button
              onClick={() => onSegment(nnInteractive)}
              disabled={isSegmentDisabled}
              variant="ghost"
              size="icon"
              className={`h-8 w-8 ${
                isSegmentDisabled
                  ? "text-slate-400 cursor-not-allowed"
                  : "text-blue-600 dark:text-blue-400 hover:bg-blue-100 dark:hover:bg-blue-900"
              }`}
              title={isLoading ? "Processing..." : "Run Segmentation"}
            >
              <Send size={16} />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
});