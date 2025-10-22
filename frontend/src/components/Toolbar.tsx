import React from "react";
import { Layers, Crosshair} from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Niivue } from "@niivue/niivue";

interface ToolbarProps {
  niivue: Niivue | null;
  onSliceClick: (niivue: Niivue | null) => void;
  onCrosshairClick: (niivue: Niivue | null) => void;
}

export const Toolbar = React.memo<ToolbarProps>(({
  niivue,
  onSliceClick,
  onCrosshairClick,
}) => (
  <div className="bg-black min-h-12 p-1 shrink-0 border-b border-slate-800">
    <div className="flex justify-center items-center gap-2 h-full">
      <Button 
        onClick={() => onSliceClick(niivue)} 
        variant="outline" 
        size="icon"
        className="h-10 w-10 border-slate-500 hover:bg-slate-700 hover:border-slate-400 text-slate-200"
        title="Toggle slice view"
      >
        <Layers size={18} />
      </Button>
      <Button
        onClick={() => onCrosshairClick(niivue)}
        variant="outline"
        size="icon"
        className="h-10 w-10 border-slate-500 hover:bg-slate-700 hover:border-slate-400 text-slate-200"
        title="Toggle crosshair"
      >
        <Crosshair size={18} />
      </Button>
    </div>
  </div>
));