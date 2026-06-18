"use client";

import { useMemo, useState } from "react";

interface LineChartProps {
  dates: string[];
  values: number[];
  height?: number;
}

/** Dependency-free responsive SVG line chart with a hover crosshair. */
export function LineChart({ dates, values, height = 220 }: LineChartProps) {
  const W = 800;
  const H = height;
  const pad = { top: 10, right: 8, bottom: 20, left: 48 };
  const [hover, setHover] = useState<number | null>(null);

  const { path, min, max, pts } = useMemo(() => {
    if (values.length === 0) return { path: "", min: 0, max: 0, pts: [] as [number, number][] };
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    const innerW = W - pad.left - pad.right;
    const innerH = H - pad.top - pad.bottom;
    const pts = values.map((v, i) => {
      const x = pad.left + (values.length === 1 ? 0 : (i / (values.length - 1)) * innerW);
      const y = pad.top + innerH - ((v - min) / span) * innerH;
      return [x, y] as [number, number];
    });
    const path = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    return { path, min, max, pts };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values, H]);

  if (values.length === 0) return <p className="text-sm text-muted-foreground">No data.</p>;

  const last = values[values.length - 1];
  const first = values[0];
  const up = last >= first;
  const stroke = up ? "var(--chart-up, #16a34a)" : "var(--chart-down, #dc2626)";
  // var() only resolves in CSS properties, not SVG presentation attributes,
  // so colors are applied via `style` below.

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    const innerW = W - pad.left - pad.right;
    const i = Math.round(((x - pad.left) / innerW) * (values.length - 1));
    setHover(Math.max(0, Math.min(values.length - 1, i)));
  };

  const fmt = (n: number) => n.toLocaleString(undefined, { maximumFractionDigits: 2 });

  return (
    <div className="w-full">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        preserveAspectRatio="none"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {[max, (max + min) / 2, min].map((v, i) => {
          const y = pad.top + (i / 2) * (H - pad.top - pad.bottom);
          return (
            <g key={i}>
              <line x1={pad.left} y1={y} x2={W - pad.right} y2={y} stroke="currentColor" opacity={0.1} />
              <text x={4} y={y + 4} fontSize={11} fill="currentColor" opacity={0.5}>
                {fmt(v)}
              </text>
            </g>
          );
        })}
        <path d={path} fill="none" style={{ stroke }} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
        {hover !== null && pts[hover] && (
          <g>
            <line
              x1={pts[hover][0]}
              y1={pad.top}
              x2={pts[hover][0]}
              y2={H - pad.bottom}
              stroke="currentColor"
              opacity={0.3}
            />
            <circle cx={pts[hover][0]} cy={pts[hover][1]} r={3} style={{ fill: stroke }} />
          </g>
        )}
      </svg>
      <div className="mt-1 flex justify-between text-xs text-muted-foreground">
        <span>{dates[0]}</span>
        <span>
          {hover !== null
            ? `${dates[hover]} · ${fmt(values[hover])}`
            : `last ${fmt(last)}`}
        </span>
        <span>{dates[dates.length - 1]}</span>
      </div>
    </div>
  );
}
