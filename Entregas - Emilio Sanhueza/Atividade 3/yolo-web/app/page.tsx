"use client";

import type { ChangeEvent, PointerEvent as ReactPointerEvent } from "react";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

const DEFAULT_MODEL_PATH = "/model_best.onnx";
const SAMPLE_MEDIA = [
  { path: "/sample_image_1.jpg", mime: "image/jpeg", kind: "image", number: 1 },
  { path: "/sample_image_2.jpg", mime: "image/jpeg", kind: "image", number: 2 },
  { path: "/sample_video_1.mp4", mime: "video/mp4", kind: "video", number: 1 },
  { path: "/sample_video_2.mp4", mime: "video/mp4", kind: "video", number: 2 },
] as const;
const DEFAULT_MODEL_SIZE = 640;
const NMS_IOU_THRESHOLD = 0.45;
const VIDEO_INFERENCE_INTERVAL_MS = 80;
const MIN_MEDIA_ZOOM = 0.5;
const MAX_MEDIA_ZOOM = 4;
const MEDIA_ZOOM_STEP = 0.15;
const DEFAULT_CLASSES = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "bus",
  "truck",
  "traffic light",
  "stop sign",
  "bench",
  "dog",
  "cat",
  "pole",
  "braille block",
  "braille line",
];
type Language = "en" | "pt";
type Status = "loading" | "ready" | "inferencing" | "error";
type MediaKind = "image" | "video" | null;
type ModelKind = "detection" | "classification";
type TensorLayout = "nchw" | "nhwc";
type Drawable = HTMLImageElement | HTMLVideoElement;
type Detection = {
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
  classId: number;
};
type Geometry = {
  originalWidth: number;
  originalHeight: number;
  scale: number;
  padX: number;
  padY: number;
};
type Classification = {
  score: number;
  classId: number;
};
type InputSpec = {
  width: number;
  height: number;
  layout: TensorLayout;
};
type TensorLike = {
  data: ArrayLike<number>;
  dims: readonly number[];
};
type SessionLike = {
  inputNames: readonly string[];
  outputNames: readonly string[];
  inputMetadata?: readonly { shape?: readonly (number | string)[] }[];
  outputMetadata?: readonly { shape?: readonly (number | string)[] }[];
  run: (feeds: Record<string, unknown>) => Promise<Record<string, TensorLike>>;
  metadata?: unknown;
  modelMetadata?: unknown;
  customMetadataMap?: unknown;
};
type OrtLike = {
  env: {
    wasm: { wasmPaths: string; numThreads: number; proxy: boolean };
  };
  Tensor: new (
    type: "float32",
    data: Float32Array,
    dimensions: readonly number[],
  ) => unknown;
  InferenceSession: {
    create: (
      model: string | ArrayBuffer,
      options: { executionProviders: readonly ["wasm"] },
    ) => Promise<SessionLike>;
  };
};
type TranslationMap = Record<number, { en: string; pt: string }>;
type ZoomAnchor = { clientX: number; clientY: number };
type PendingZoomAnchor = {
  contentRatioX: number;
  contentRatioY: number;
  viewportX: number;
  viewportY: number;
};
type MediaFrameSize = { width: number; height: number };
type PanStart = {
  pointerId: number;
  x: number;
  y: number;
  scrollLeft: number;
  scrollTop: number;
};

const normalizeClassName = (name: string) =>
  name.trim().toLowerCase().replace(/[_-]+/g, " ").replace(/\s+/g, " ");

const CLASS_TRANSLATIONS: Record<string, { en: string; pt: string }> = {
  barricade: { en: "barricade", pt: "barricada" },
  bench: { en: "bench", pt: "banco" },
  bicycle: { en: "bicycle", pt: "bicicleta" },
  bollard: { en: "bollard", pt: "balizador" },
  bus: { en: "bus", pt: "ônibus" },
  car: { en: "car", pt: "carro" },
  carrier: { en: "carrier", pt: "carrinho de carga" },
  cat: { en: "cat", pt: "gato" },
  chair: { en: "chair", pt: "cadeira" },
  dog: { en: "dog", pt: "cachorro" },
  "fire hydrant": { en: "fire hydrant", pt: "hidrante" },
  kiosk: { en: "kiosk", pt: "quiosque" },
  motorcycle: { en: "motorcycle", pt: "motocicleta" },
  "movable signage": { en: "movable signage", pt: "sinalização móvel" },
  "parking meter": { en: "parking meter", pt: "parquímetro" },
  person: { en: "person", pt: "pessoa" },
  pole: { en: "pole", pt: "poste" },
  "potted plant": { en: "potted plant", pt: "planta em vaso" },
  "power controller": {
    en: "power controller",
    pt: "controlador de energia",
  },
  scooter: { en: "scooter", pt: "patinete" },
  stop: { en: "stop sign", pt: "placa de pare" },
  "stop sign": { en: "stop sign", pt: "placa de pare" },
  stroller: { en: "stroller", pt: "carrinho de bebê" },
  table: { en: "table", pt: "mesa" },
  "traffic light": { en: "traffic light", pt: "semáforo" },
  "traffic light controller": {
    en: "traffic light controller",
    pt: "controlador de semáforo",
  },
  "traffic sign": { en: "traffic sign", pt: "placa de trânsito" },
  "tree trunk": { en: "tree trunk", pt: "tronco de árvore" },
  truck: { en: "truck", pt: "caminhão" },
  wheelchair: { en: "wheelchair", pt: "cadeira de rodas" },
  "braille block": {
    en: "warning tactile paving",
    pt: "piso tátil de alerta",
  },
  "braille line": {
    en: "directional tactile paving",
    pt: "linha de piso tátil",
  },
  bache: { en: "pothole", pt: "buraco" },
  desgaste: { en: "surface wear", pt: "desgaste" },
  longitudinal: {
    en: "longitudinal crack",
    pt: "fissura longitudinal",
  },
  cocodrilo: { en: "alligator crack", pt: "trinca couro de crocodilo" },
  transversal: { en: "transverse crack", pt: "fissura transversal" },
};

const translationFor = (name: string) => {
  const normalized = normalizeClassName(name);
  const generic = /^class (\d+)$/.exec(normalized);
  if (generic) {
    return { en: "class " + generic[1], pt: "classe " + generic[1] };
  }
  return CLASS_TRANSLATIONS[normalized] ?? {
    en: normalized,
    pt: normalized,
  };
};

const localizedClassText = (
  names: readonly string[],
  translations: TranslationMap,
  language: Language,
) =>
  names.map((name, index) =>
    translations[index]?.[language] ?? translationFor(name)[language]
  ).join(", ");

const positiveDimension = (value: number | string | undefined) => {
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isInteger(numeric) && numeric > 0 ? numeric : null;
};

const readInputSpec = (session: SessionLike): InputSpec => {
  const shape = session.inputMetadata?.[0]?.shape;
  if (!shape) {
    return {
      width: DEFAULT_MODEL_SIZE,
      height: DEFAULT_MODEL_SIZE,
      layout: "nchw",
    };
  }
  if (shape.length !== 4) {
    throw new Error(
      "Unsupported model input shape: [" + shape.join(", ") + "]",
    );
  }
  const channelsFirst = positiveDimension(shape[1]) === 3;
  const channelsLast = positiveDimension(shape[3]) === 3;
  if (!channelsFirst && !channelsLast) {
    throw new Error(
      "The model input must have 3 RGB channels in NCHW or NHWC layout.",
    );
  }
  if (channelsLast) {
    return {
      height: positiveDimension(shape[1]) ?? DEFAULT_MODEL_SIZE,
      width: positiveDimension(shape[2]) ?? DEFAULT_MODEL_SIZE,
      layout: "nhwc",
    };
  }
  return {
    height: positiveDimension(shape[2]) ?? DEFAULT_MODEL_SIZE,
    width: positiveDimension(shape[3]) ?? DEFAULT_MODEL_SIZE,
    layout: "nchw",
  };
};

const readModelKind = (session: SessionLike): ModelKind => {
  const shape = session.outputMetadata?.[0]?.shape;
  return shape?.length === 2 ? "classification" : "detection";
};

const readClassificationCount = (session: SessionLike) => {
  const shape = session.outputMetadata?.[0]?.shape;
  return shape?.length === 2 ? positiveDimension(shape[1]) : null;
};

const yieldToUi = () =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });

const parseClasses = (value: string) =>
  value.split(",").map((name) => name.trim()).filter(Boolean);

const clampMediaZoom = (value: number) =>
  Math.min(MAX_MEDIA_ZOOM, Math.max(MIN_MEDIA_ZOOM, value));

const parseMetadataValue = (value: unknown): string[] | null => {
  if (Array.isArray(value)) {
    const names = value.map(String).map((name) => name.trim()).filter(Boolean);
    return names.length ? names : null;
  }
  if (typeof value === "object" && value !== null) {
    const names = Object.entries(value as Record<string, unknown>)
      .filter(([key]) => /^\d+$/.test(key))
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([, name]) => String(name).trim())
      .filter(Boolean);
    return names.length ? names : null;
  }
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }
  try {
    const parsed = JSON.parse(value) as unknown;
    const names = parseMetadataValue(parsed);
    if (names) {
      return names;
    }
  } catch {
    const entries: Array<{ id: number; name: string }> = [];
    const pattern = /['"]?(\d+)['"]?\s*:\s*['"]([^'"]+)['"]/g;
    let match = pattern.exec(value);
    while (match) {
      entries.push({ id: Number(match[1]), name: match[2].trim() });
      match = pattern.exec(value);
    }
    if (entries.length) {
      return entries
        .sort((left, right) => left.id - right.id)
        .map((entry) => entry.name);
    }
  }
  const commaNames = parseClasses(value);
  return commaNames.length > 1 ? commaNames : null;
};

type VarintResult = { value: number; nextOffset: number };

const readVarint = (
  bytes: Uint8Array,
  startOffset: number,
  limit: number,
): VarintResult | null => {
  let value = 0;
  let multiplier = 1;
  let offset = startOffset;
  while (offset < limit && multiplier <= 2 ** 49) {
    const byte = bytes[offset];
    offset += 1;
    value += (byte & 0x7f) * multiplier;
    if ((byte & 0x80) === 0) {
      return { value, nextOffset: offset };
    }
    multiplier *= 128;
  }
  return null;
};

const skipProtobufField = (
  bytes: Uint8Array,
  offset: number,
  limit: number,
  wireType: number,
) => {
  if (wireType === 0) {
    return readVarint(bytes, offset, limit)?.nextOffset ?? null;
  }
  if (wireType === 1) {
    return offset + 8 <= limit ? offset + 8 : null;
  }
  if (wireType === 2) {
    const length = readVarint(bytes, offset, limit);
    if (!length || length.nextOffset + length.value > limit) {
      return null;
    }
    return length.nextOffset + length.value;
  }
  if (wireType === 5) {
    return offset + 4 <= limit ? offset + 4 : null;
  }
  return null;
};

const readMetadataEntry = (
  bytes: Uint8Array,
  startOffset: number,
  limit: number,
) => {
  const decoder = new TextDecoder();
  let key = "";
  let value = "";
  let offset = startOffset;
  while (offset < limit) {
    const tag = readVarint(bytes, offset, limit);
    if (!tag) {
      break;
    }
    offset = tag.nextOffset;
    const fieldNumber = Math.floor(tag.value / 8);
    const wireType = tag.value & 7;
    if ((fieldNumber === 1 || fieldNumber === 2) && wireType === 2) {
      const length = readVarint(bytes, offset, limit);
      if (!length || length.nextOffset + length.value > limit) {
        break;
      }
      const text = decoder.decode(
        bytes.subarray(length.nextOffset, length.nextOffset + length.value),
      );
      if (fieldNumber === 1) {
        key = text;
      } else {
        value = text;
      }
      offset = length.nextOffset + length.value;
      continue;
    }
    const nextOffset = skipProtobufField(bytes, offset, limit, wireType);
    if (nextOffset === null) {
      break;
    }
    offset = nextOffset;
  }
  return { key, value };
};

const readOnnxMetadataClasses = (model: ArrayBuffer): string[] | null => {
  const bytes = new Uint8Array(model);
  const metadata: Record<string, string> = {};
  let offset = 0;
  while (offset < bytes.length) {
    const tag = readVarint(bytes, offset, bytes.length);
    if (!tag) {
      break;
    }
    offset = tag.nextOffset;
    const fieldNumber = Math.floor(tag.value / 8);
    const wireType = tag.value & 7;
    if (fieldNumber === 14 && wireType === 2) {
      const length = readVarint(bytes, offset, bytes.length);
      if (!length || length.nextOffset + length.value > bytes.length) {
        break;
      }
      const entryEnd = length.nextOffset + length.value;
      const entry = readMetadataEntry(bytes, length.nextOffset, entryEnd);
      if (entry.key) {
        metadata[entry.key] = entry.value;
      }
      offset = entryEnd;
      continue;
    }
    const nextOffset = skipProtobufField(
      bytes,
      offset,
      bytes.length,
      wireType,
    );
    if (nextOffset === null) {
      break;
    }
    offset = nextOffset;
  }
  return parseMetadataValue(metadata.names) ??
    parseMetadataValue(metadata.classes) ??
    parseMetadataValue(metadata.class_names);
};

const readMetadataClasses = (session: SessionLike): string[] | null => {
  const containers = [
    session.metadata,
    session.modelMetadata,
    session.customMetadataMap,
  ];
  for (const container of containers) {
    if (typeof container !== "object" || container === null) {
      continue;
    }
    const record = container as Record<string, unknown>;
    const candidates = [
      record.names,
      record.classes,
      record.class_names,
      record.customMetadataMap,
    ];
    for (const candidate of candidates) {
      const direct = parseMetadataValue(candidate);
      if (direct) {
        return direct;
      }
      if (typeof candidate === "object" && candidate !== null) {
        const nested = candidate as Record<string, unknown>;
        const nestedNames = parseMetadataValue(nested.names) ??
          parseMetadataValue(nested.classes) ??
          parseMetadataValue(nested.class_names);
        if (nestedNames) {
          return nestedNames;
        }
      }
    }
  }
  return null;
};

const dimensionsOf = (source: Drawable) =>
  source instanceof HTMLVideoElement
    ? { width: source.videoWidth, height: source.videoHeight }
    : { width: source.naturalWidth, height: source.naturalHeight };

const iou = (left: Detection, right: Detection) => {
  const width = Math.max(
    0,
    Math.min(left.x + left.width, right.x + right.width) -
      Math.max(left.x, right.x),
  );
  const height = Math.max(
    0,
    Math.min(left.y + left.height, right.y + right.height) -
      Math.max(left.y, right.y),
  );
  const intersection = width * height;
  const union = left.width * left.height +
    right.width * right.height -
    intersection;
  return union > 0 ? intersection / union : 0;
};

const nms = async (detections: Detection[]) => {
  const pending = detections.slice().sort((a, b) => b.score - a.score);
  const selected: Detection[] = [];
  while (pending.length) {
    const candidate = pending.shift();
    if (!candidate) {
      break;
    }
    selected.push(candidate);
    for (let index = pending.length - 1; index >= 0; index -= 1) {
      if (
        pending[index].classId === candidate.classId &&
        iou(candidate, pending[index]) > NMS_IOU_THRESHOLD
      ) {
        pending.splice(index, 1);
      }
    }
    if (selected.length % 32 === 0) {
      await yieldToUi();
    }
  }
  return selected;
};

const decodeOutput = async (
  output: TensorLike,
  geometry: Geometry,
  threshold: number,
) => {
  if (output.dims.length !== 3 || Number(output.dims[0]) !== 1) {
    throw new Error(
      "Unsupported output shape: [" + Array.from(output.dims).join(", ") + "]",
    );
  }
  const channelCount = Number(output.dims[1]);
  const anchorCount = Number(output.dims[2]);
  const classCount = channelCount - 4;
  if (classCount < 1 || anchorCount < 1) {
    throw new Error("The YOLO output has no valid classes or anchors.");
  }
  const data = output.data;
  const candidates: Detection[] = [];
  for (let anchor = 0; anchor < anchorCount; anchor += 1) {
    const xCenter = Number(data[0 * anchorCount + anchor]);
    const yCenter = Number(data[1 * anchorCount + anchor]);
    const modelWidth = Number(data[2 * anchorCount + anchor]);
    const modelHeight = Number(data[3 * anchorCount + anchor]);
    let bestScore = Number.NEGATIVE_INFINITY;
    let bestClass = -1;
    for (let classId = 0; classId < classCount; classId += 1) {
      const score = Number(data[(4 + classId) * anchorCount + anchor]);
      if (score > bestScore) {
        bestScore = score;
        bestClass = classId;
      }
    }
    if (bestClass < 0 || !Number.isFinite(bestScore) || bestScore < threshold) {
      continue;
    }
    const rawX = (xCenter - modelWidth / 2 - geometry.padX) / geometry.scale;
    const rawY = (yCenter - modelHeight / 2 - geometry.padY) / geometry.scale;
    const rawWidth = modelWidth / geometry.scale;
    const rawHeight = modelHeight / geometry.scale;
    const x1 = Math.max(0, Math.min(geometry.originalWidth, rawX));
    const y1 = Math.max(0, Math.min(geometry.originalHeight, rawY));
    const x2 = Math.max(
      0,
      Math.min(geometry.originalWidth, rawX + rawWidth),
    );
    const y2 = Math.max(
      0,
      Math.min(geometry.originalHeight, rawY + rawHeight),
    );
    if (x2 > x1 && y2 > y1) {
      candidates.push({
        x: x1,
        y: y1,
        width: x2 - x1,
        height: y2 - y1,
        score: bestScore,
        classId: bestClass,
      });
    }
    if (anchor > 0 && anchor % 512 === 0) {
      await yieldToUi();
    }
  }
  return nms(candidates);
};

const decodeClassificationOutput = (output: TensorLike) => {
  if (
    output.dims.length !== 2 ||
    Number(output.dims[0]) !== 1 ||
    Number(output.dims[1]) < 1
  ) {
    throw new Error(
      "Unsupported classification output shape: [" +
        Array.from(output.dims).join(", ") + "]",
    );
  }
  const raw = Array.from(output.data, Number);
  if (raw.some((value) => !Number.isFinite(value))) {
    throw new Error("The classifier returned non-finite scores.");
  }
  const sum = raw.reduce((total, value) => total + value, 0);
  const alreadyProbabilities = raw.every((value) => value >= 0 && value <= 1) &&
    Math.abs(sum - 1) < 0.02;
  const probabilities = alreadyProbabilities
    ? raw
    : (() => {
      const maximum = Math.max(...raw);
      const exponentials = raw.map((value) => Math.exp(value - maximum));
      const total = exponentials.reduce((current, value) => current + value, 0);
      return exponentials.map((value) => value / total);
    })();
  return probabilities
    .map((score, classId) => ({ score, classId }))
    .sort((left, right) => right.score - left.score)
    .slice(0, 5);
};

const readModelFile = (file: File) =>
  new Promise<ArrayBuffer>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) {
        resolve(reader.result);
      } else {
        reject(new Error("The selected model could not be read."));
      }
    };
    reader.onerror = () =>
      reject(reader.error ?? new Error("Failed to read the local model."));
    reader.readAsArrayBuffer(file);
  });

export default function Page() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const mediaFrameRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const ortRef = useRef<OrtLike | null>(null);
  const sessionRef = useRef<SessionLike | null>(null);
  const boxesRef = useRef<Detection[]>([]);
  const classificationsRef = useRef<Classification[]>([]);
  const inputSpecRef = useRef<InputSpec>({
    width: DEFAULT_MODEL_SIZE,
    height: DEFAULT_MODEL_SIZE,
    layout: "nchw",
  });
  const modelKindRef = useRef<ModelKind>("detection");
  const translationsRef = useRef<TranslationMap>({});
  const translationCacheRef = useRef(
    new Map<string, { en: string; pt: string }>(),
  );
  const classesRef = useRef(DEFAULT_CLASSES);
  const languageRef = useRef<Language>("en");
  const confidenceRef = useRef(0.4);
  const mediaZoomRef = useRef(1);
  const pendingZoomAnchorRef = useRef<PendingZoomAnchor | null>(null);
  const panStartRef = useRef<PanStart | null>(null);
  const mediaKindRef = useRef<MediaKind>(null);
  const animationFrameRef = useRef<number | null>(null);
  const videoFrameRef = useRef<number | null>(null);
  const lastInferenceRef = useRef(0);
  const inferenceBusyRef = useRef(false);
  const objectUrlRef = useRef<string | null>(null);
  const loadSequenceRef = useRef(0);
  const classEditVersionRef = useRef(0);
  const imageTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const redrawRef = useRef<() => void>(() => undefined);

  const [status, setStatus] = useState<Status>("loading");
  const [statusDetail, setStatusDetail] = useState("Loading ONNX Runtime");
  const [modelName, setModelName] = useState("Default model");
  const [modelKind, setModelKind] = useState<ModelKind>("detection");
  const [sourceClasses, setSourceClasses] = useState<string[]>(DEFAULT_CLASSES);
  const [classText, setClassText] = useState(DEFAULT_CLASSES.join(", "));
  const [language, setLanguage] = useState<Language>("en");
  const [confidence, setConfidence] = useState(0.4);
  const [hasMedia, setHasMedia] = useState(false);
  const [activeMediaKind, setActiveMediaKind] = useState<MediaKind>(null);
  const [mediaZoom, setMediaZoom] = useState(1);
  const [mediaFrameSize, setMediaFrameSize] = useState<MediaFrameSize>({
    width: 0,
    height: 0,
  });
  const [isPanning, setIsPanning] = useState(false);

  const applyMediaZoom = useCallback((value: number, anchor?: ZoomAnchor) => {
    const nextZoom = Math.round(clampMediaZoom(value) * 100) / 100;
    if (nextZoom === mediaZoomRef.current) {
      return;
    }
    const viewport = viewportRef.current;
    if (viewport) {
      const bounds = viewport.getBoundingClientRect();
      const viewportX = anchor
        ? anchor.clientX - bounds.left
        : viewport.clientWidth / 2;
      const viewportY = anchor
        ? anchor.clientY - bounds.top
        : viewport.clientHeight / 2;
      pendingZoomAnchorRef.current = {
        contentRatioX: (viewport.scrollLeft + viewportX) /
          Math.max(1, viewport.scrollWidth),
        contentRatioY: (viewport.scrollTop + viewportY) /
          Math.max(1, viewport.scrollHeight),
        viewportX,
        viewportY,
      };
    }
    mediaZoomRef.current = nextZoom;
    setMediaZoom(nextZoom);
  }, []);

  const adjustMediaZoom = useCallback(
    (amount: number) => applyMediaZoom(mediaZoomRef.current + amount),
    [applyMediaZoom],
  );

  const releaseObjectUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  const stopVideo = useCallback(() => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (videoFrameRef.current !== null && videoRef.current) {
      videoRef.current.cancelVideoFrameCallback(videoFrameRef.current);
      videoFrameRef.current = null;
    }
    videoRef.current?.pause();
    inferenceBusyRef.current = false;
    lastInferenceRef.current = 0;
  }, []);

  const draw = useCallback((providedSource?: Drawable) => {
    const source = providedSource ??
      (mediaKindRef.current === "video" ? videoRef.current : imageRef.current);
    const canvas = canvasRef.current;
    if (!source || !canvas) {
      return;
    }
    const dimensions = dimensionsOf(source);
    if (dimensions.width < 1 || dimensions.height < 1) {
      return;
    }
    if (
      canvas.width !== dimensions.width ||
      canvas.height !== dimensions.height
    ) {
      canvas.width = dimensions.width;
      canvas.height = dimensions.height;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    context.clearRect(0, 0, canvas.width, canvas.height);
    // Video stays on its own composited layer. The canvas only paints the
    // overlay, so inference work can never reduce the media playback rate.
    if (!(source instanceof HTMLVideoElement)) {
      context.drawImage(source, 0, 0, canvas.width, canvas.height);
    }
    context.lineWidth = Math.max(2, canvas.width / 640);
    context.textBaseline = "middle";
    context.font = Math.max(13, Math.round(canvas.width / 70)) +
      "px ui-monospace, SFMono-Regular, Menlo, monospace";
    const classifications = classificationsRef.current;
    if (classifications.length) {
      const visible = classifications.filter(
        (result) => result.score >= confidenceRef.current,
      );
      const results = visible.length ? visible : classifications.slice(0, 1);
      const padding = 8;
      const rowHeight = Math.max(26, Math.round(canvas.width / 42));
      results.forEach((result, index) => {
        const translated = translationsRef.current[result.classId];
        const fallback = classesRef.current[result.classId] ??
          "class " + result.classId;
        const name = translated?.[languageRef.current] ?? fallback;
        const label = name + " " + (result.score * 100).toFixed(1) + "%";
        const color = "hsl(" +
          ((result.classId * 67 + 38) % 360) +
          " 78% 62%)";
        const width = context.measureText(label).width + padding * 2;
        const x = padding;
        const y = padding + index * (rowHeight + 5);
        context.fillStyle = color;
        context.fillRect(x, y, width, rowHeight);
        context.fillStyle = "#282828";
        context.fillText(label, x + padding, y + rowHeight / 2);
      });
    }
    for (const box of boxesRef.current) {
      if (box.score < confidenceRef.current) {
        continue;
      }
      const color = "hsl(" + ((box.classId * 67 + 38) % 360) + " 78% 62%)";
      const translated = translationsRef.current[box.classId];
      const fallback = classesRef.current[box.classId] ??
        "class " + box.classId;
      const name = translated?.[languageRef.current] ?? fallback;
      const label = name + " " + (box.score * 100).toFixed(1) + "%";
      const padding = 7;
      const pillHeight = Math.max(22, Math.round(canvas.width / 48));
      const pillWidth = context.measureText(label).width + padding * 2;
      const pillX = Math.max(0, Math.min(canvas.width - pillWidth, box.x));
      const pillY = Math.max(0, box.y - pillHeight);
      context.strokeStyle = color;
      context.strokeRect(box.x, box.y, box.width, box.height);
      context.fillStyle = color;
      context.fillRect(pillX, pillY, pillWidth, pillHeight);
      context.fillStyle = "#282828";
      context.fillText(label, pillX + padding, pillY + pillHeight / 2);
    }
  }, []);

  useEffect(() => {
    redrawRef.current = () => draw();
  }, [draw]);

  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    const anchor = pendingZoomAnchorRef.current;
    if (!viewport || !anchor) {
      return;
    }
    pendingZoomAnchorRef.current = null;
    viewport.scrollLeft =
      anchor.contentRatioX * viewport.scrollWidth - anchor.viewportX;
    viewport.scrollTop =
      anchor.contentRatioY * viewport.scrollHeight - anchor.viewportY;
  }, [mediaZoom]);

  useEffect(() => {
    const frame = mediaFrameRef.current;
    if (!frame || !hasMedia) {
      return;
    }
    const updateFrameSize = () => {
      const nextSize = {
        width: frame.offsetWidth,
        height: frame.offsetHeight,
      };
      setMediaFrameSize((current) =>
        current.width === nextSize.width && current.height === nextSize.height
          ? current
          : nextSize
      );
    };
    updateFrameSize();
    const observer = new ResizeObserver(updateFrameSize);
    observer.observe(frame);
    return () => observer.disconnect();
  }, [activeMediaKind, hasMedia]);

  useEffect(() => {
    const handleZoomShortcut = (event: KeyboardEvent) => {
      if (!hasMedia || (!event.ctrlKey && !event.metaKey)) {
        return;
      }
      if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        adjustMediaZoom(MEDIA_ZOOM_STEP);
      } else if (event.key === "-") {
        event.preventDefault();
        adjustMediaZoom(-MEDIA_ZOOM_STEP);
      } else if (event.key === "0") {
        event.preventDefault();
        applyMediaZoom(1);
      }
    };
    window.addEventListener("keydown", handleZoomShortcut);
    return () => window.removeEventListener("keydown", handleZoomShortcut);
  }, [adjustMediaZoom, applyMediaZoom, hasMedia]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }
    let pinchStartDistance = 0;
    let pinchStartZoom = 1;
    const touchDistance = (touches: TouchList) => {
      const horizontal = touches[0].clientX - touches[1].clientX;
      const vertical = touches[0].clientY - touches[1].clientY;
      return Math.hypot(horizontal, vertical);
    };
    const handleWheel = (event: WheelEvent) => {
      if (!hasMedia || (!event.ctrlKey && !event.metaKey)) {
        return;
      }
      event.preventDefault();
      const factor = Math.exp(-event.deltaY * 0.002);
      applyMediaZoom(mediaZoomRef.current * factor, {
        clientX: event.clientX,
        clientY: event.clientY,
      });
    };
    const handleTouchStart = (event: TouchEvent) => {
      if (!hasMedia || event.touches.length !== 2) {
        return;
      }
      event.preventDefault();
      pinchStartDistance = touchDistance(event.touches);
      pinchStartZoom = mediaZoomRef.current;
    };
    const handleTouchMove = (event: TouchEvent) => {
      if (event.touches.length !== 2 || pinchStartDistance === 0) {
        return;
      }
      event.preventDefault();
      const midpoint = {
        clientX: (event.touches[0].clientX + event.touches[1].clientX) / 2,
        clientY: (event.touches[0].clientY + event.touches[1].clientY) / 2,
      };
      applyMediaZoom(
        pinchStartZoom * (touchDistance(event.touches) / pinchStartDistance),
        midpoint,
      );
    };
    const handleTouchEnd = () => {
      pinchStartDistance = 0;
    };
    viewport.addEventListener("wheel", handleWheel, { passive: false });
    viewport.addEventListener("touchstart", handleTouchStart, {
      passive: false,
    });
    viewport.addEventListener("touchmove", handleTouchMove, {
      passive: false,
    });
    viewport.addEventListener("touchend", handleTouchEnd);
    viewport.addEventListener("touchcancel", handleTouchEnd);
    return () => {
      viewport.removeEventListener("wheel", handleWheel);
      viewport.removeEventListener("touchstart", handleTouchStart);
      viewport.removeEventListener("touchmove", handleTouchMove);
      viewport.removeEventListener("touchend", handleTouchEnd);
      viewport.removeEventListener("touchcancel", handleTouchEnd);
    };
  }, [applyMediaZoom, hasMedia]);

  const preprocess = useCallback(
    async (source: Drawable, ort: OrtLike) => {
      const dimensions = dimensionsOf(source);
      if (dimensions.width < 1 || dimensions.height < 1) {
        throw new Error("The selected media has no valid dimensions.");
      }
      await yieldToUi();
      let offscreen = offscreenRef.current;
      if (!offscreen) {
        offscreen = document.createElement("canvas");
        offscreenRef.current = offscreen;
      }
      const input = inputSpecRef.current;
      offscreen.width = input.width;
      offscreen.height = input.height;
      const context = offscreen.getContext("2d", { willReadFrequently: true });
      if (!context) {
        throw new Error("Could not create the preprocessing canvas.");
      }
      const scale = Math.min(
        input.width / dimensions.width,
        input.height / dimensions.height,
      );
      let padX = 0;
      let padY = 0;
      if (modelKindRef.current === "classification") {
        context.drawImage(source, 0, 0, input.width, input.height);
      } else {
        const scaledWidth = dimensions.width * scale;
        const scaledHeight = dimensions.height * scale;
        padX = (input.width - scaledWidth) / 2;
        padY = (input.height - scaledHeight) / 2;
        context.fillStyle = "#727272";
        context.fillRect(0, 0, input.width, input.height);
        context.drawImage(source, padX, padY, scaledWidth, scaledHeight);
      }
      await yieldToUi();
      const pixels = context.getImageData(
        0,
        0,
        input.width,
        input.height,
      ).data;
      const plane = input.width * input.height;
      const tensorData = new Float32Array(plane * 3);
      for (let pixel = 0; pixel < plane; pixel += 1) {
        const rgba = pixel * 4;
        if (input.layout === "nhwc") {
          const rgb = pixel * 3;
          tensorData[rgb] = pixels[rgba] / 255;
          tensorData[rgb + 1] = pixels[rgba + 1] / 255;
          tensorData[rgb + 2] = pixels[rgba + 2] / 255;
        } else {
          tensorData[pixel] = pixels[rgba] / 255;
          tensorData[plane + pixel] = pixels[rgba + 1] / 255;
          tensorData[plane * 2 + pixel] = pixels[rgba + 2] / 255;
        }
        if (pixel > 0 && pixel % 65536 === 0) {
          await yieldToUi();
        }
      }
      const tensorDimensions = input.layout === "nhwc"
        ? [1, input.height, input.width, 3]
        : [1, 3, input.height, input.width];
      return {
        tensor: new ort.Tensor(
          "float32",
          tensorData,
          tensorDimensions,
        ),
        geometry: {
          originalWidth: dimensions.width,
          originalHeight: dimensions.height,
          scale,
          padX,
          padY,
        } satisfies Geometry,
      };
    },
    [],
  );

  const infer = useCallback(
    async (source: Drawable, updateStatus: boolean) => {
      const ort = ortRef.current;
      const session = sessionRef.current;
      if (!ort || !session) {
        return;
      }
      if (updateStatus) {
        setStatus("inferencing");
        setStatusDetail("Inferencing image");
      }
      try {
        const threshold = confidenceRef.current;
        const prepared = await preprocess(source, ort);
        await yieldToUi();
        const feeds: Record<string, unknown> = {};
        feeds[session.inputNames[0]] = prepared.tensor;
        const outputs = await session.run(feeds);
        await yieldToUi();
        const output = outputs[session.outputNames[0]];
        if (!output) {
          throw new Error("The model returned no output tensor.");
        }
        const classification = output.dims.length === 2
          ? decodeClassificationOutput(output)
          : null;
        const detections = classification
          ? null
          : await decodeOutput(output, prepared.geometry, threshold);
        if (
          sessionRef.current !== session ||
          (updateStatus && imageRef.current !== source)
        ) {
          return;
        }
        classificationsRef.current = classification ?? [];
        boxesRef.current = detections ?? [];
        draw(source);
        if (updateStatus) {
          setStatus("ready");
          setStatusDetail("Model Ready");
        }
      } catch (error) {
        console.error(error);
        if (updateStatus) {
          setStatus("error");
          setStatusDetail(
            error instanceof Error ? error.message : "Inference failed",
          );
        }
      }
    },
    [draw, preprocess],
  );

  const startVideo = useCallback(() => {
    const video = videoRef.current;
    if (!video) {
      return;
    }
    const renderFrame = (timestamp: number) => {
      if (mediaKindRef.current !== "video") {
        return;
      }
      if (
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        timestamp - lastInferenceRef.current >=
          VIDEO_INFERENCE_INTERVAL_MS &&
        !inferenceBusyRef.current &&
        sessionRef.current
      ) {
        lastInferenceRef.current = timestamp;
        inferenceBusyRef.current = true;
        void infer(video, false).finally(() => {
          inferenceBusyRef.current = false;
        });
      }
      scheduleNextFrame();
    };
    const scheduleNextFrame = () => {
      if (typeof video.requestVideoFrameCallback === "function") {
        videoFrameRef.current = video.requestVideoFrameCallback(
          (timestamp) => renderFrame(timestamp),
        );
      } else {
        animationFrameRef.current = requestAnimationFrame(renderFrame);
      }
    };
    lastInferenceRef.current = 0;
    draw(video);
    scheduleNextFrame();
  }, [draw, infer]);

  const loadModel = useCallback(
    async (source: string | ArrayBuffer, label: string) => {
      const ort = ortRef.current;
      if (!ort) {
        throw new Error("ONNX Runtime is not loaded.");
      }
      const sequence = loadSequenceRef.current + 1;
      loadSequenceRef.current = sequence;
      boxesRef.current = [];
      classificationsRef.current = [];
      redrawRef.current();
      setStatus("loading");
      setStatusDetail("Loading model");
      try {
        await yieldToUi();
        let model = source;
        if (typeof source === "string") {
          const response = await fetch(source);
          if (!response.ok) {
            throw new Error("Model download failed: " + response.status);
          }
          model = await response.arrayBuffer();
        }
        if (sequence !== loadSequenceRef.current) {
          return;
        }
        const embeddedClasses = model instanceof ArrayBuffer
          ? readOnnxMetadataClasses(model)
          : null;
        const session = await ort.InferenceSession.create(model, {
          executionProviders: ["wasm"],
        });
        if (sequence !== loadSequenceRef.current) {
          return;
        }
        sessionRef.current = session;
        const detectedKind = readModelKind(session);
        inputSpecRef.current = readInputSpec(session);
        modelKindRef.current = detectedKind;
        setModelKind(detectedKind);
        const classificationCount = readClassificationCount(session);
        const fallbackClasses = detectedKind === "classification" &&
            classificationCount
          ? Array.from(
            { length: classificationCount },
            (_, index) => "class " + index,
          )
          : DEFAULT_CLASSES;
        const names = embeddedClasses ??
          readMetadataClasses(session) ??
          fallbackClasses;
        setSourceClasses(names);
        setModelName(label);
        const videoActive = mediaKindRef.current === "video";
        setStatus(videoActive ? "inferencing" : "ready");
        setStatusDetail(videoActive ? "Inferencing video" : "Model Ready");
        if (mediaKindRef.current === "image" && imageRef.current) {
          await infer(imageRef.current, true);
        }
      } catch (error) {
        console.error(error);
        setStatus("error");
        setStatusDetail(
          error instanceof Error ? error.message : "Model loading failed",
        );
      }
    },
    [infer],
  );

  const loadImage = useCallback(
    (url: string, objectUrl: boolean) => {
      stopVideo();
      applyMediaZoom(1);
      releaseObjectUrl();
      if (objectUrl) {
        objectUrlRef.current = url;
      }
      boxesRef.current = [];
      classificationsRef.current = [];
      imageRef.current = null;
      mediaKindRef.current = "image";
      setActiveMediaKind("image");
      setHasMedia(true);
      const image = new Image();
      image.decoding = "async";
      image.onload = () => {
        imageRef.current = image;
        draw(image);
        void infer(image, true);
      };
      image.onerror = () => {
        setStatus("error");
        setStatusDetail("Image decoding failed");
      };
      image.src = url;
    },
    [applyMediaZoom, draw, infer, releaseObjectUrl, stopVideo],
  );

  const loadVideo = useCallback(
    (url: string, objectUrl: boolean) => {
      stopVideo();
      applyMediaZoom(1);
      releaseObjectUrl();
      if (objectUrl) {
        objectUrlRef.current = url;
      }
      boxesRef.current = [];
      classificationsRef.current = [];
      imageRef.current = null;
      mediaKindRef.current = "video";
      setActiveMediaKind("video");
      setHasMedia(true);
      const video = videoRef.current;
      if (!video) {
        setStatus("error");
        setStatusDetail("Video element unavailable");
        return;
      }
      video.src = url;
      video.onloadedmetadata = () => {
        setStatus("inferencing");
        setStatusDetail("Inferencing video");
        void video.play().then(startVideo).catch((error: unknown) => {
          console.error(error);
          setStatus("error");
          setStatusDetail("Video playback was blocked");
        });
      };
      video.onerror = () => {
        setStatus("error");
        setStatusDetail("Video decoding failed");
      };
      video.load();
    },
    [applyMediaZoom, releaseObjectUrl, startVideo, stopVideo],
  );

  const loadMedia = useCallback(
    (url: string, mime: string, objectUrl: boolean) => {
      if (mime.startsWith("video/")) {
        loadVideo(url, objectUrl);
      } else {
        loadImage(url, objectUrl);
      }
    },
    [loadImage, loadVideo],
  );

  useEffect(() => {
    let cancelled = false;
    const initialize = async () => {
      try {
        const imported = await import("onnxruntime-web");
        if (cancelled) {
          return;
        }
        const ort = imported as unknown as OrtLike;
        ort.env.wasm.wasmPaths =
          "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.proxy = true;
        ortRef.current = ort;
        await loadModel(DEFAULT_MODEL_PATH, "Default model");
        if (!cancelled && mediaKindRef.current === null) {
          loadImage(SAMPLE_MEDIA[1].path, false);
        }
      } catch (error) {
        console.error(error);
        if (!cancelled) {
          setStatus("error");
          setStatusDetail(
            error instanceof Error
              ? error.message
              : "ONNX Runtime initialization failed",
          );
        }
      }
    };
    void initialize();
    return () => {
      cancelled = true;
      loadSequenceRef.current += 1;
      stopVideo();
      releaseObjectUrl();
      if (imageTimerRef.current !== null) {
        clearTimeout(imageTimerRef.current);
      }
    };
  }, [loadImage, loadModel, releaseObjectUrl, stopVideo]);

  useEffect(() => {
    const translationVersion = classEditVersionRef.current + 1;
    classEditVersionRef.current = translationVersion;
    classesRef.current = sourceClasses.length ? sourceClasses : DEFAULT_CLASSES;
    const initial: TranslationMap = {};
    classesRef.current.forEach((name, index) => {
      const normalized = normalizeClassName(name);
      initial[index] = translationCacheRef.current.get(normalized) ??
        translationFor(name);
    });
    translationsRef.current = initial;
    setClassText(
      localizedClassText(classesRef.current, initial, languageRef.current),
    );
    redrawRef.current();

    const controller = new AbortController();
    const timer = setTimeout(() => {
      const unknownIndexes = classesRef.current.map((name, index) => {
        const normalized = normalizeClassName(name);
        return CLASS_TRANSLATIONS[normalized] ||
            /^class \d+$/.test(normalized) ||
            translationCacheRef.current.has(normalized)
          ? -1
          : index;
      }).filter((index) => index >= 0);
      if (!unknownIndexes.length) {
        return;
      }
      const pendingClasses = unknownIndexes.map((classIndex) => ({
        classIndex,
        name: classesRef.current[classIndex],
      }));
      const translate = async (name: string, target: Language) => {
        const endpoint = "https://api.mymemory.translated.net/get?q=" +
          encodeURIComponent(normalizeClassName(name)) +
          "&langpair=autodetect|" + target;
        const response = await fetch(endpoint, { signal: controller.signal });
        if (!response.ok) {
          throw new Error("Translation request failed: " + response.status);
        }
        const payload = await response.json() as {
          responseData?: { translatedText?: string };
        };
        const encoded = payload.responseData?.translatedText?.trim();
        const decoder = document.createElement("textarea");
        decoder.innerHTML = encoded ?? "";
        const translated = decoder.value.trim();
        return translated || normalizeClassName(name);
      };
      const translateOne = async (name: string) => {
        const fallback = translationFor(name);
        const translateOrFallback = async (target: Language) => {
          try {
            return await translate(name, target);
          } catch (error) {
            if (
              error instanceof DOMException && error.name === "AbortError"
            ) {
              throw error;
            }
            console.warn(
              "Class translation unavailable for " + name + " (" + target + ")",
              error,
            );
            return fallback[target];
          }
        };
        const [en, pt] = await Promise.all([
          translateOrFallback("en"),
          translateOrFallback("pt"),
        ]);
        return { en, pt };
      };
      const translateAll = async () => {
        const translated = new Map<number, { en: string; pt: string }>();
        let cursor = 0;
        const worker = async () => {
          while (cursor < pendingClasses.length) {
            const position = cursor;
            cursor += 1;
            const { classIndex, name } = pendingClasses[position];
            const result = await translateOne(name);
            translated.set(classIndex, result);
            translationCacheRef.current.set(normalizeClassName(name), result);
          }
        };
        const workerCount = Math.min(4, pendingClasses.length);
        await Promise.all(Array.from({ length: workerCount }, () => worker()));
        return translated;
      };
      void translateAll()
        .then((translated) => {
          if (
            controller.signal.aborted ||
            classEditVersionRef.current !== translationVersion
          ) {
            return;
          }
          const updated = { ...translationsRef.current };
          translated.forEach((value, classIndex) => {
            updated[classIndex] = value;
          });
          translationsRef.current = updated;
          setClassText(
            localizedClassText(
              classesRef.current,
              updated,
              languageRef.current,
            ),
          );
          redrawRef.current();
        })
        .catch((error: unknown) => {
          if (
            !(error instanceof DOMException && error.name === "AbortError")
          ) {
            console.warn("Class translation unavailable", error);
          }
        });
    }, 450);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [sourceClasses]);

  useEffect(() => {
    languageRef.current = language;
    setClassText(
      localizedClassText(
        classesRef.current,
        translationsRef.current,
        language,
      ),
    );
    redrawRef.current();
  }, [language]);

  const changeClasses = (event: ChangeEvent<HTMLTextAreaElement>) => {
    classEditVersionRef.current += 1;
    const value = event.target.value;
    setClassText(value);
    const edited = parseClasses(value);
    if (!edited.length) {
      return;
    }
    const updated: TranslationMap = {};
    edited.forEach((name, index) => {
      const previous = translationsRef.current[index] ?? translationFor(name);
      updated[index] = { ...previous, [language]: name };
    });
    translationsRef.current = updated;
    classesRef.current = edited.map(
      (name, index) => updated[index]?.en ?? name,
    );
    redrawRef.current();
  };

  const uploadModel = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    await loadModel(await readModelFile(file), file.name);
    event.target.value = "";
  };

  const uploadMedia = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    loadMedia(URL.createObjectURL(file), file.type, true);
    event.target.value = "";
  };

  const chooseSample = (sample: (typeof SAMPLE_MEDIA)[number]) => {
    loadMedia(sample.path, sample.mime, false);
  };

  const changeConfidence = (event: ChangeEvent<HTMLInputElement>) => {
    const value = Number(event.target.value);
    confidenceRef.current = value;
    setConfidence(value);
    redrawRef.current();
    if (mediaKindRef.current === "image" && imageRef.current) {
      if (imageTimerRef.current !== null) {
        clearTimeout(imageTimerRef.current);
      }
      imageTimerRef.current = setTimeout(() => {
        if (imageRef.current) {
          void infer(imageRef.current, true);
        }
      }, 100);
    }
  };

  const startPanning = (event: ReactPointerEvent<HTMLDivElement>) => {
    const viewport = viewportRef.current;
    if (
      !viewport ||
      !hasMedia ||
      mediaZoom <= 1 ||
      event.pointerType === "touch" ||
      event.button !== 0
    ) {
      return;
    }
    event.preventDefault();
    viewport.setPointerCapture(event.pointerId);
    panStartRef.current = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      scrollLeft: viewport.scrollLeft,
      scrollTop: viewport.scrollTop,
    };
    setIsPanning(true);
  };

  const continuePanning = (event: ReactPointerEvent<HTMLDivElement>) => {
    const viewport = viewportRef.current;
    const start = panStartRef.current;
    if (!viewport || !start || start.pointerId !== event.pointerId) {
      return;
    }
    viewport.scrollLeft = start.scrollLeft - (event.clientX - start.x);
    viewport.scrollTop = start.scrollTop - (event.clientY - start.y);
  };

  const stopPanning = (event: ReactPointerEvent<HTMLDivElement>) => {
    const viewport = viewportRef.current;
    if (panStartRef.current?.pointerId !== event.pointerId) {
      return;
    }
    if (viewport?.hasPointerCapture(event.pointerId)) {
      viewport.releasePointerCapture(event.pointerId);
    }
    panStartRef.current = null;
    setIsPanning(false);
  };

  const statusStyles: Record<
    Status,
    { label: string; style: string; dot: string }
  > = {
    loading: {
      label: "Loading",
      style: "border-[#d79921]/50 bg-[#d79921]/15 text-[#fabd2f]",
      dot: "bg-[#fabd2f]",
    },
    ready: {
      label: "Model Ready",
      style: "border-[#98971a]/50 bg-[#98971a]/15 text-[#b8bb26]",
      dot: "bg-[#b8bb26]",
    },
    inferencing: {
      label: "Inferencing",
      style: "border-[#458588]/50 bg-[#458588]/15 text-[#83a598]",
      dot: "bg-[#83a598]",
    },
    error: {
      label: "Error",
      style: "border-[#cc241d]/50 bg-[#cc241d]/15 text-[#fb4934]",
      dot: "bg-[#fb4934]",
    },
  };
  const statusView = statusStyles[status];
  const text = language === "en"
    ? {
      model: "Model",
      localModel: "Custom ONNX model",
      reset: "Reset to Default Model",
      classes: "Class Manager",
      hint: "Comma-separated class names",
      language: "Language",
      confidence: "Confidence",
      media: "Media",
      upload: "Upload image or video",
      samples: "Sample media",
      sampleImage: "Image",
      sampleVideo: "Video",
      viewport: "Detection viewport",
      classificationViewport: "Classification viewport",
      zoomIn: "Zoom in",
      zoomOut: "Zoom out",
      resetZoom: "Reset zoom",
      zoomHelp: "Pinch or Ctrl/⌘ +/− to zoom · drag to move",
      empty: "Select an image, video, or sample to begin detection.",
      defaultModel: "Default",
    }
    : {
      model: "Modelo",
      localModel: "Modelo ONNX personalizado",
      reset: "Restaurar Modelo Padrão",
      classes: "Gerenciador de Classes",
      hint: "Nomes das classes separados por vírgulas",
      language: "Idioma",
      confidence: "Confiança",
      media: "Mídia",
      upload: "Enviar imagem ou vídeo",
      samples: "Mídias de exemplo",
      sampleImage: "Imagem",
      sampleVideo: "Vídeo",
      viewport: "Área de detecção",
      classificationViewport: "Área de classificação",
      zoomIn: "Ampliar",
      zoomOut: "Reduzir",
      resetZoom: "Restaurar zoom",
      zoomHelp: "Use pinça ou Ctrl/⌘ +/− para ampliar · arraste para mover",
      empty: "Selecione uma imagem, vídeo ou exemplo para iniciar.",
      defaultModel: "Padrão",
    };

  return (
    <main className="min-h-screen bg-[#1d2021] text-[#ebdbb2]">
      <div className="mx-auto flex min-h-screen max-w-[1800px] flex-col">
        <header className="flex flex-col gap-4 border-b border-[#504945] bg-[#282828] px-5 py-5 shadow-xl shadow-black/20 sm:flex-row sm:items-center sm:justify-between lg:px-8">
          <div>
            <p className="mb-1 font-mono text-xs uppercase tracking-[0.28em] text-[#d79921]">
              Browser-native computer vision
            </p>
            <h1 className="text-2xl font-bold text-[#fbf1c7] sm:text-3xl">
              YONO{" "}
              <span className="font-normal text-[#a89984]">
                (You Only Need ONNX)
              </span>
            </h1>
          </div>
          <div className="flex min-w-0 items-center gap-3">
            <span
              className={"inline-flex shrink-0 items-center gap-2 rounded-full border px-3 py-1.5 font-mono text-xs font-semibold " +
                statusView.style}
            >
              <span
                className={"h-2 w-2 rounded-full " +
                  statusView.dot +
                  (status === "loading" || status === "inferencing"
                    ? " animate-pulse"
                    : "")}
              />
              {statusView.label}
            </span>
            <span
              className="max-w-[260px] truncate text-xs text-[#a89984]"
              title={statusDetail}
            >
              {statusDetail}
            </span>
          </div>
        </header>

        <div className="grid flex-1 grid-cols-1 lg:grid-cols-[340px_minmax(0,1fr)]">
          <aside className="border-b border-[#504945] bg-[#282828] p-5 lg:border-b-0 lg:border-r lg:p-6">
            <div className="space-y-7">
              <section>
                <div className="mb-3 flex items-center justify-between gap-3">
                  <h2 className="text-sm font-semibold uppercase tracking-wider text-[#fabd2f]">
                    {text.model}
                  </h2>
                  <span
                    className="max-w-[170px] truncate rounded bg-[#3c3836] px-2 py-1 font-mono text-[10px] text-[#bdae93]"
                    title={modelName}
                  >
                    {modelName === "Default model"
                      ? text.defaultModel
                      : modelName}
                  </span>
                </div>
                <label className="block cursor-pointer rounded-lg border border-dashed border-[#665c54] bg-[#32302f] px-4 py-4 text-center text-sm text-[#d5c4a1] transition hover:border-[#d79921]">
                  {text.localModel}
                  <input
                    type="file"
                    accept=".onnx,application/octet-stream"
                    className="sr-only"
                    onChange={(event) => {
                      void uploadModel(event);
                    }}
                  />
                </label>
                <button
                  type="button"
                  onClick={() => {
                    if (ortRef.current) {
                      void loadModel(DEFAULT_MODEL_PATH, "Default model");
                    }
                  }}
                  className="mt-2 w-full rounded-lg border border-[#665c54] px-3 py-2 text-sm font-medium transition hover:border-[#83a598] hover:text-[#83a598]"
                >
                  {text.reset}
                </button>
              </section>

              <section>
                <label
                  htmlFor="classes"
                  className="mb-2 block text-sm font-semibold uppercase tracking-wider text-[#fabd2f]"
                >
                  {text.classes}
                </label>
                <textarea
                  id="classes"
                  value={classText}
                  onChange={changeClasses}
                  rows={5}
                  spellCheck={false}
                  className="w-full resize-y rounded-lg border border-[#504945] bg-[#1d2021] p-3 font-mono text-xs leading-5 outline-none focus:border-[#d79921] focus:ring-2 focus:ring-[#d79921]/20"
                />
                <p className="mt-1.5 text-xs text-[#928374]">{text.hint}</p>
              </section>

              <section className="flex items-center justify-between gap-4">
                <span className="text-sm font-semibold uppercase tracking-wider text-[#fabd2f]">
                  {text.language}
                </span>
                <div className="flex rounded-full border border-[#504945] bg-[#1d2021] p-1">
                  {(["en", "pt"] as const).map((option) => (
                    <button
                      key={option}
                      type="button"
                      aria-pressed={language === option}
                      onClick={() => setLanguage(option)}
                      className={"rounded-full px-3 py-1 font-mono text-xs font-bold transition " +
                        (language === option
                          ? "bg-[#d79921] text-[#282828]"
                          : "text-[#a89984] hover:text-[#ebdbb2]")}
                    >
                      {option.toUpperCase()}
                    </button>
                  ))}
                </div>
              </section>

              <section>
                <div className="mb-2 flex items-center justify-between">
                  <label
                    htmlFor="confidence"
                    className="text-sm font-semibold uppercase tracking-wider text-[#fabd2f]"
                  >
                    {text.confidence}
                  </label>
                  <output
                    htmlFor="confidence"
                    className="rounded bg-[#3c3836] px-2 py-1 font-mono text-xs text-[#b8bb26]"
                  >
                    {(confidence * 100).toFixed(0)}%
                  </output>
                </div>
                <input
                  id="confidence"
                  type="range"
                  min="0.01"
                  max="1"
                  step="0.01"
                  value={confidence}
                  onChange={changeConfidence}
                  className="h-2 w-full cursor-pointer appearance-none rounded-full bg-[#504945] accent-[#b8bb26]"
                />
              </section>

              <section>
                <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-[#fabd2f]">
                  {text.media}
                </h2>
                <label className="block cursor-pointer rounded-lg bg-[#458588] px-4 py-3 text-center text-sm font-bold text-[#fbf1c7] transition hover:bg-[#83a598] hover:text-[#282828]">
                  {text.upload}
                  <input
                    type="file"
                    accept="image/*,video/*"
                    className="sr-only"
                    onChange={uploadMedia}
                  />
                </label>
                <p className="mb-2 mt-4 text-xs uppercase tracking-wider text-[#928374]">
                  {text.samples}
                </p>
                <div className="grid grid-cols-3 gap-2">
                  {SAMPLE_MEDIA.map((sample) => (
                    <button
                      key={sample.path}
                      type="button"
                      title={sample.path}
                      onClick={() => chooseSample(sample)}
                      className="rounded-lg border border-[#504945] bg-[#32302f] px-2 py-3 font-mono text-xs text-[#d5c4a1] transition hover:border-[#b8bb26] hover:text-[#b8bb26]"
                    >
                      {sample.kind === "video"
                        ? text.sampleVideo + " " + sample.number
                        : text.sampleImage + " " + sample.number}
                    </button>
                  ))}
                </div>
              </section>
            </div>
          </aside>

          <section className="flex min-h-[560px] min-w-0 flex-col p-4 sm:p-6 lg:p-8">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-[#d5c4a1]">
                {modelKind === "classification"
                  ? text.classificationViewport
                  : text.viewport}
              </h2>
              <div className="flex items-center gap-3">
                <span className="hidden font-mono text-xs text-[#7c6f64] sm:inline">
                  {modelKind === "classification"
                    ? "WASM worker · softmax"
                    : "WASM worker · NMS 0.45"}
                </span>
                <div
                  className="flex items-center rounded-lg border border-[#504945] bg-[#282828] p-1"
                  aria-label="Media zoom controls"
                >
                  <button
                    type="button"
                    aria-label={text.zoomOut}
                    title={text.zoomOut + " (Ctrl/⌘ −)"}
                    disabled={!hasMedia || mediaZoom <= MIN_MEDIA_ZOOM}
                    onClick={() => adjustMediaZoom(-MEDIA_ZOOM_STEP)}
                    className="flex h-7 w-7 items-center justify-center rounded text-lg leading-none text-[#d5c4a1] transition hover:bg-[#3c3836] hover:text-[#fabd2f] disabled:cursor-not-allowed disabled:opacity-30"
                  >
                    −
                  </button>
                  <button
                    type="button"
                    title={text.resetZoom + " (Ctrl/⌘ 0)"}
                    disabled={!hasMedia}
                    onClick={() => applyMediaZoom(1)}
                    className="min-w-14 rounded px-1.5 py-1 font-mono text-[11px] text-[#bdae93] transition hover:bg-[#3c3836] hover:text-[#fbf1c7] disabled:cursor-not-allowed disabled:opacity-30"
                  >
                    {Math.round(mediaZoom * 100)}%
                  </button>
                  <button
                    type="button"
                    aria-label={text.zoomIn}
                    title={text.zoomIn + " (Ctrl/⌘ +)"}
                    disabled={!hasMedia || mediaZoom >= MAX_MEDIA_ZOOM}
                    onClick={() => adjustMediaZoom(MEDIA_ZOOM_STEP)}
                    className="flex h-7 w-7 items-center justify-center rounded text-lg leading-none text-[#d5c4a1] transition hover:bg-[#3c3836] hover:text-[#fabd2f] disabled:cursor-not-allowed disabled:opacity-30"
                  >
                    +
                  </button>
                </div>
              </div>
            </div>
            <div
              ref={viewportRef}
              onPointerDown={startPanning}
              onPointerMove={continuePanning}
              onPointerUp={stopPanning}
              onPointerCancel={stopPanning}
              onLostPointerCapture={stopPanning}
              className={
                "relative min-h-[480px] flex-1 overflow-auto overscroll-contain rounded-xl border border-[#504945] bg-[#0d0e0f] shadow-2xl shadow-black/30 " +
                (hasMedia && mediaZoom > 1
                  ? isPanning
                    ? "cursor-grabbing select-none"
                    : "cursor-grab"
                  : "")
              }
              style={{ touchAction: "pan-x pan-y" }}
            >
              {!hasMedia && (
                <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center p-8 text-center">
                  <div>
                    <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-[#504945] bg-[#282828] font-mono text-xl font-bold text-[#d79921]">
                      YN
                    </div>
                    <p className="max-w-md text-sm leading-6 text-[#928374]">
                      {text.empty}
                    </p>
                  </div>
                </div>
              )}
              {hasMedia && (
                <div className="pointer-events-none sticky left-0 top-3 z-20 flex h-0 w-full justify-center px-3">
                  <span className="rounded-full border border-white/10 bg-[#1d2021]/85 px-3 py-1.5 font-mono text-[10px] text-[#bdae93] shadow-lg shadow-black/30 backdrop-blur-sm sm:text-[11px]">
                    {text.zoomHelp}
                  </span>
                </div>
              )}
              <div
                className="grid min-h-full min-w-full place-items-center p-4"
                style={{
                  width: mediaFrameSize.width
                    ? `max(100%, ${Math.ceil(mediaFrameSize.width * mediaZoom) + 32}px)`
                    : "100%",
                  height: mediaFrameSize.height
                    ? `max(100%, ${Math.ceil(mediaFrameSize.height * mediaZoom) + 32}px)`
                    : "100%",
                }}
              >
                <div
                  ref={mediaFrameRef}
                  className="relative inline-flex max-h-[calc(100vh-190px)] max-w-full items-center justify-center will-change-transform"
                  style={{ transform: `scale(${mediaZoom})` }}
                >
                  <video
                    ref={videoRef}
                    className={activeMediaKind === "video"
                      ? "block max-h-[calc(100vh-190px)] max-w-full object-contain"
                      : "hidden"}
                    muted
                    loop
                    playsInline
                    preload="auto"
                  />
                  <canvas
                    ref={canvasRef}
                    width={640}
                    height={480}
                    aria-label={modelKind === "classification"
                      ? "Image classification output"
                      : "YOLO object detection output"}
                    className={
                      (activeMediaKind === "video"
                        ? "pointer-events-none absolute inset-0 h-full w-full"
                        : "block max-h-[calc(100vh-190px)] max-w-full object-contain") +
                      (hasMedia ? " opacity-100" : " opacity-0")
                    }
                    style={activeMediaKind === "video"
                      ? undefined
                      : { width: "auto", height: "auto" }}
                  />
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
