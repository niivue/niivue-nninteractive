import { useState, useCallback, useMemo } from "react";
import { DRAG_MODE, SLICE_TYPE, type Niivue } from "@niivue/niivue";

export const useToolbarState = () => {
  const [sliceType, setSliceType] = useState<SLICE_TYPE>(SLICE_TYPE.AXIAL);

  const sliceOptions = useMemo(() => [
    SLICE_TYPE.AXIAL,
    SLICE_TYPE.CORONAL,
    SLICE_TYPE.SAGITTAL,
    SLICE_TYPE.MULTIPLANAR,
  ], []);

  const handleSliceClick = useCallback((niivue: Niivue | null) => {
    if (!niivue) return;

    const currentIndex = sliceOptions.indexOf(sliceType);
    const nextIndex = (currentIndex + 1) % sliceOptions.length;
    const nextSliceType = sliceOptions[nextIndex];
    
    setSliceType(nextSliceType);
    niivue.setSliceType(nextSliceType);
  }, [sliceType, sliceOptions]);

  const handleCrosshairClick = useCallback((niivue: Niivue | null) => {
    if (!niivue) return;
    
    const width = niivue.opts.crosshairWidth;
    niivue.setCrosshairWidth(width > 0 ? 0 : 1);
  }, []);

  const handleMeasureClick = useCallback((niivue: Niivue | null) => {
    if (!niivue) return;
    
    const mode = niivue.opts.dragMode;
    niivue.setDragMode(
      mode === DRAG_MODE.measurement
        ? DRAG_MODE.contrast
        : DRAG_MODE.measurement
    );
  }, []);

  return {
    sliceType,
    handleSliceClick,
    handleCrosshairClick,
    handleMeasureClick,
  };
};