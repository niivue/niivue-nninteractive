import React from "react";

interface StatusDisplayProps {
  userId: string | null;
}

export const StatusDisplay = React.memo<StatusDisplayProps>(({ userId }) => {
  if (!userId) return null;

  return (
    <div className="text-center py-2 bg-slate-900 border-t border-slate-700">
      <p className="text-xs text-slate-400 font-mono">
        Session ID: <span className="text-slate-300">{userId}</span>
      </p>
    </div>
  );
});