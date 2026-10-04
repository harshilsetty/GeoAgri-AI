"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Circle, CircleMarker, MapContainer, Marker, Polygon, TileLayer, Tooltip, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

type MapPickerProps = {
  lat: number;
  lon: number;
  onChange: (lat: number, lon: number) => void;
  areaRadiusMeters?: number;
  showRadiusArea?: boolean;
  polygonVertices?: Array<[number, number]>;
  sampledPoints?: Array<[number, number]>;
};

type SearchResult = {
  display_name: string;
  lat: string;
  lon: string;
};

// Fix default marker icons when bundling with Next.js.
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

function RecenterMap({ lat, lon }: { lat: number; lon: number }) {
  const map = useMap();
  const previousRef = useRef<[number, number] | null>(null);

  useEffect(() => {
    const previous = previousRef.current;
    const next: [number, number] = [lat, lon];
    if (!previous) {
      previousRef.current = next;
      map.setView(next, map.getZoom(), { animate: false });
      return;
    }

    const latDiff = Math.abs(previous[0] - lat);
    const lonDiff = Math.abs(previous[1] - lon);
    if (latDiff < 1e-7 && lonDiff < 1e-7) {
      return;
    }

    previousRef.current = next;
    map.flyTo(next, map.getZoom(), { animate: true, duration: 0.45 });
  }, [lat, lon, map]);

  return null;
}

function ClickHandler({ onChange }: { onChange: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(event) {
      onChange(event.latlng.lat, event.latlng.lng);
    },
  });
  return null;
}

export default function MapPicker({
  lat,
  lon,
  onChange,
  areaRadiusMeters = 300,
  showRadiusArea = false,
  polygonVertices = [],
  sampledPoints = [],
}: MapPickerProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const usePermanentPointLabels = true;
  const showPointLabels = sampledPoints.length > 0;

  const markerPosition = useMemo(() => [lat, lon] as [number, number], [lat, lon]);
  const activeMarkerIcon = useMemo(
    () =>
      L.divIcon({
        className: "geo-pulse-marker",
        html: '<span class="geo-pulse-dot"></span>',
        iconSize: [22, 22],
        iconAnchor: [11, 11],
      }),
    []
  );

  async function handleSearch() {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    setSearching(true);
    try {
      const url = `https://nominatim.openstreetmap.org/search?format=json&limit=5&q=${encodeURIComponent(query.trim())}`;
      const response = await fetch(url, {
        headers: {
          Accept: "application/json",
        },
      });
      const data = (await response.json()) as SearchResult[];
      setResults(Array.isArray(data) ? data : []);
    } catch {
      setResults([]);
    } finally {
      setSearching(false);
    }
  }

  return (
    <div className="space-y-2">
      <p className="text-xs uppercase tracking-wide text-dim">Select Location on Map</p>
      <div className="flex gap-2">
        <input
          className="input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search location"
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              handleSearch();
            }
          }}
        />
        <button type="button" className="button-primary px-3 py-2" onClick={handleSearch}>
          {searching ? "..." : "Go"}
        </button>
      </div>

      {results.length > 0 ? (
        <div className="max-h-32 overflow-auto rounded-xl border border-borderSoft bg-slate-900/80">
          {results.map((item) => (
            <button
              key={`${item.lat}-${item.lon}-${item.display_name}`}
              type="button"
              className="block w-full border-b border-borderSoft px-3 py-2 text-left text-xs text-dim hover:bg-white/5"
              onClick={() => {
                onChange(Number(item.lat), Number(item.lon));
                setResults([]);
                setQuery(item.display_name);
              }}
            >
              {item.display_name}
            </button>
          ))}
        </div>
      ) : null}

      <div className="overflow-hidden rounded-xl border border-borderSoft transition hover:border-cyan-300/35">
        <MapContainer center={markerPosition} zoom={10} scrollWheelZoom preferCanvas className="h-[220px] w-full">
          <TileLayer
            attribution="Tiles &copy; Esri"
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />
          {showRadiusArea ? (
            <Circle
              center={markerPosition}
              radius={areaRadiusMeters}
              pathOptions={{ color: "#22d3ee", weight: 2, fillOpacity: 0.12 }}
            />
          ) : null}
          {polygonVertices.length >= 3 ? (
            <Polygon
              positions={polygonVertices}
              pathOptions={{ color: "#f59e0b", weight: 2, fillOpacity: 0.12 }}
            />
          ) : null}
          {polygonVertices.map((vertex, index) => (
            <CircleMarker
              key={`vertex-${index}-${vertex[0]}-${vertex[1]}`}
              center={vertex}
              radius={5}
              pathOptions={{ color: "#f59e0b", fillColor: "#f59e0b", fillOpacity: 0.9, weight: 1 }}
            />
          ))}
          {sampledPoints.map((point, index) => (
            <CircleMarker
              key={`sample-${index}-${point[0]}-${point[1]}`}
              center={point}
              radius={4}
              pathOptions={{ color: "#10b981", fillColor: "#10b981", fillOpacity: 0.85, weight: 1 }}
            >
              {showPointLabels ? (
                <Tooltip
                  direction="top"
                  offset={[0, -8]}
                  opacity={1}
                  permanent={usePermanentPointLabels}
                  className="leaf-point-label"
                >
                  {index + 1}
                </Tooltip>
              ) : null}
            </CircleMarker>
          ))}
          <Marker position={markerPosition} icon={activeMarkerIcon} />
          <ClickHandler onChange={onChange} />
          <RecenterMap lat={lat} lon={lon} />
        </MapContainer>
      </div>
      <p className="text-xs text-dim">{lat.toFixed(6)}, {lon.toFixed(6)}</p>
    </div>
  );
}
