"use client"

import { useEffect, useRef, useState } from "react"
import Image from "next/image"
import { ClipboardPaste, Download, ImagePlus, Loader2, X } from "lucide-react"
import * as XLSX from "xlsx"
import { Button } from "@/components/ui/button"
import { apiFetch } from "@/lib/api"

type ExtractedTable = {
  status: "ready" | "needs_review" | "cannot_convert"
  reason: string
  columns: Array<{ name: string; kind: "text" | "number" }>
  rows: Array<Array<string | null>>
  uncertain_cells: Array<{ row: number; column: number }>
}

type PhotoSource = {
  id: string
  file: File
  mimeType: string
  url: string
  result?: ExtractedTable
  error?: string
}

const IMAGE_TYPES = new Map([
  ["png", "image/png"], ["jpg", "image/jpeg"], ["jpeg", "image/jpeg"], ["webp", "image/webp"],
])
const MAX_PHOTOS = 10
const MAX_PHOTO_BYTES = 10 * 1024 * 1024
const ACCEPTED_IMAGES = "image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp"
const CLIPBOARD_EXTENSIONS = new Map([
  ["image/png", "png"], ["image/jpeg", "jpg"], ["image/webp", "webp"],
])

function sourceStatus(source: PhotoSource, activeId: string | null) {
  if (source.id === activeId) return "Extracting…"
  if (source.error) return "Try again"
  if (source.result?.status === "ready") return "Ready"
  if (source.result?.status === "needs_review") return "Needs review"
  if (source.result?.status === "cannot_convert") return "Can’t convert"
  return "Queued"
}

function excelCell(value: string | null, kind: "text" | "number") {
  const text = value ?? ""
  if (kind === "number" && /^-?(0|[1-9]\d*)(\.\d+)?$/.test(text) && text.replace(/\D/g, "").length <= 15) {
    return Number(text)
  }
  return text
}

function downloadWorkbook(sources: PhotoSource[]) {
  const workbook = XLSX.utils.book_new()
  sources.forEach((source, index) => {
    const result = source.result
    if (!result || result.status === "cannot_convert") return
    const headers = result.columns.map((column, columnIndex) => column.name.trim() || `Column ${columnIndex + 1}`)
    const rows = result.rows.map((row) => row.map((value, columnIndex) => excelCell(value, result.columns[columnIndex].kind)))
    const stem = source.file.name.replace(/\.[^.]+$/, "").replace(/[\\/?*:[\]]/g, " ").trim()
    const sheetName = `${index + 1} - ${stem || "Photo"}`.slice(0, 31).replace(/'+$/, "")
    XLSX.utils.book_append_sheet(workbook, XLSX.utils.aoa_to_sheet([headers, ...rows]), sheetName)
  })
  if (workbook.SheetNames.length) XLSX.writeFile(workbook, "photo-tables.xlsx")
}

export function PhotoImport({ onClose }: { onClose: () => void }) {
  const dialogRef = useRef<HTMLDialogElement>(null)
  const pickerRef = useRef<HTMLInputElement>(null)
  const objectUrls = useRef(new Set<string>())
  const request = useRef<AbortController | null>(null)
  const pastedCount = useRef(0)
  const [sources, setSources] = useState<PhotoSource[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [activeId, setActiveId] = useState<string | null>(null)
  const [isExtracting, setIsExtracting] = useState(false)
  const [uploadError, setUploadError] = useState("")

  useEffect(() => {
    const dialog = dialogRef.current
    if (dialog && !dialog.open) dialog.showModal()
    const urls = objectUrls.current
    return () => {
      request.current?.abort()
      for (const url of urls) URL.revokeObjectURL(url)
      urls.clear()
    }
  }, [])

  const selected = sources.find((source) => source.id === selectedId) ?? sources[0]
  const result = selected?.result
  const unclearCells = new Set(result?.uncertain_cells.map(({ row, column }) => `${row}:${column}`))
  const ready = sources.filter((source) => source.result && source.result.status !== "cannot_convert")
  const unclearCount = ready.reduce((count, source) => count + (source.result?.uncertain_cells.length ?? 0), 0)
  const pending = sources.filter((source) => !source.result)

  function addFiles(files: File[]) {
    if (isExtracting) return
    const added: PhotoSource[] = []
    const rejected: string[] = []
    for (const file of files) {
      const mimeType = IMAGE_TYPES.get(file.name.split(".").pop()?.toLowerCase() ?? "")
      if (!mimeType || file.size === 0 || file.size > MAX_PHOTO_BYTES || objectUrls.current.size >= MAX_PHOTOS) {
        rejected.push(file.name)
        continue
      }
      const url = URL.createObjectURL(file)
      objectUrls.current.add(url)
      added.push({ id: crypto.randomUUID(), file, mimeType, url })
    }
    if (added.length) {
      setSources((current) => [...current, ...added])
      setSelectedId((current) => current ?? added[0].id)
    }
    setUploadError(rejected.length ? `Couldn’t add ${rejected.join(", ")}. Use up to ${MAX_PHOTOS} PNG, JPEG, or WebP photos, each 10 MB or smaller.` : "")
  }

  function removeSource(source: PhotoSource) {
    URL.revokeObjectURL(source.url)
    objectUrls.current.delete(source.url)
    const remaining = sources.filter((item) => item.id !== source.id)
    setSources(remaining)
    if (selected?.id === source.id) setSelectedId(remaining[0]?.id ?? null)
    setUploadError("")
  }

  function addPastedImages(images: Blob[]) {
    const files = images.flatMap((blob) => {
      const extension = CLIPBOARD_EXTENSIONS.get(blob.type)
      if (!extension) return []
      return [new File([blob], `Pasted photo ${++pastedCount.current}.${extension}`, { type: blob.type })]
    })
    if (files.length) addFiles(files)
    else setUploadError("The clipboard has no PNG, JPEG, or WebP image.")
  }

  async function pasteFromClipboard() {
    try {
      const clipboard = await navigator.clipboard.read()
      const images = await Promise.all(clipboard.flatMap((item) => {
        const type = item.types.find((mimeType) => CLIPBOARD_EXTENSIONS.has(mimeType))
        return type ? [item.getType(type)] : []
      }))
      if (dialogRef.current?.open && !request.current) addPastedImages(images)
    } catch {
      if (dialogRef.current?.open) {
        setUploadError("Copy an image, then press Ctrl+V or ⌘V inside this dialog.")
      }
    }
  }

  async function extractTables() {
    if (request.current || !pending.length) return
    const controller = new AbortController()
    request.current = controller
    setIsExtracting(true)
    setUploadError("")
    try {
      for (const source of pending) {
        if (controller.signal.aborted) break
        setActiveId(source.id)
        setSources((current) => current.map((item) => item.id === source.id ? { ...item, error: undefined } : item))
        try {
          const response = await apiFetch("/dataset/extract-image", {
            method: "POST",
            headers: { "Content-Type": source.mimeType },
            body: source.file,
            signal: controller.signal,
          })
          if (!response.ok) {
            const body = await response.json().catch(() => null)
            throw new Error(typeof body?.detail === "string" ? body.detail : "This photo could not be processed.")
          }
          const extracted: ExtractedTable = await response.json()
          if (!controller.signal.aborted) {
            setSources((current) => current.map((item) => item.id === source.id ? { ...item, result: extracted } : item))
          }
        } catch (error) {
          if (!controller.signal.aborted) {
            const message = error instanceof Error ? error.message : "This photo could not be processed."
            setSources((current) => current.map((item) => item.id === source.id ? { ...item, error: message } : item))
          }
        }
      }
    } finally {
      request.current = null
      if (!controller.signal.aborted) {
        setActiveId(null)
        setIsExtracting(false)
      }
    }
  }

  function editResult(update: (table: ExtractedTable) => ExtractedTable) {
    setSources((current) => current.map((source) =>
      source.id === selected?.id && source.result ? { ...source, result: update(source.result) } : source
    ))
  }

  return (
    <dialog
      ref={dialogRef}
      onClose={onClose}
      onPaste={(event) => {
        const clipboard = event.clipboardData
        const images = Array.from(clipboard.files).filter((file) => file.type.startsWith("image/"))
        if (!images.length) return
        if (event.target instanceof HTMLInputElement && clipboard.types.includes("text/plain")) return
        event.preventDefault()
        if (!isExtracting) addPastedImages(images)
      }}
      aria-labelledby="photo-import-title"
      className="fixed inset-0 m-auto h-[min(84dvh,760px)] w-[min(96vw,1080px)] max-w-none overflow-hidden rounded-2xl border border-border bg-background text-foreground shadow-2xl backdrop:bg-black/65"
    >
      <div
        className="flex h-full flex-col"
        onDragOver={(event) => {
          if (event.dataTransfer.types.includes("Files")) event.preventDefault()
        }}
        onDrop={(event) => {
          if (!event.dataTransfer.types.includes("Files")) return
          event.preventDefault()
          addFiles(Array.from(event.dataTransfer.files))
        }}
      >
        <header className="flex items-start justify-between gap-3 border-b border-border px-5 py-4">
          <div>
            <h2 id="photo-import-title" className="text-lg font-semibold">Photo to Excel</h2>
            <p className="text-sm text-muted-foreground">Extract paper records, check the table, and download one worksheet per photo.</p>
          </div>
          <Button variant="ghost" size="icon" aria-label="Close photo import" onClick={() => dialogRef.current?.close()}>
            <X aria-hidden="true" />
          </Button>
        </header>

        <div className="grid min-h-0 flex-1 grid-rows-[auto_minmax(0,1fr)] sm:grid-cols-[15rem_minmax(0,1fr)] sm:grid-rows-1">
          <aside aria-label="Photo sources" className="max-h-40 min-w-0 overflow-y-auto border-b border-border bg-card/45 p-3 sm:max-h-none sm:border-r sm:border-b-0">
            <div className="mb-2 flex items-center justify-between gap-2">
              <h3 className="text-sm font-semibold">Sources <span className="text-xs text-muted-foreground">{sources.length}</span></h3>
              <Button variant="outline" size="sm" disabled={isExtracting || sources.length >= MAX_PHOTOS} onClick={() => pickerRef.current?.click()}>
                <ImagePlus aria-hidden="true" /> Add photos
              </Button>
              <input
                ref={pickerRef}
                hidden
                type="file"
                multiple
                accept={ACCEPTED_IMAGES}
                onChange={(event) => {
                  addFiles(Array.from(event.currentTarget.files ?? []))
                  event.currentTarget.value = ""
                }}
              />
            </div>
            {uploadError && <p role="alert" className="mb-2 break-words text-xs text-destructive">{uploadError}</p>}
            <Button variant="ghost" size="sm" className="mb-2 w-full justify-start"
              disabled={isExtracting || sources.length >= MAX_PHOTOS} onClick={() => void pasteFromClipboard()}>
              <ClipboardPaste aria-hidden="true" /> Paste image
            </Button>
            {!sources.length && <p className="rounded-lg border border-dashed border-border p-3 text-xs text-muted-foreground">Drop photos here, use Add photos, or paste with Ctrl+V / ⌘V.</p>}
            <ul className="space-y-1.5">
              {sources.map((source) => (
                <li key={source.id} className={`flex min-w-0 items-center gap-1 rounded-lg border ${selected?.id === source.id ? "border-primary/40 bg-primary/10" : "border-transparent hover:bg-accent/60"}`}>
                  <button type="button" aria-pressed={selected?.id === source.id} onClick={() => setSelectedId(source.id)} className="flex min-w-0 flex-1 items-center gap-2 p-1.5 text-left">
                    <Image src={source.url} width={36} height={36} alt="" unoptimized className="size-9 shrink-0 rounded border border-border bg-muted object-cover" />
                    <span className="min-w-0">
                      <span className="block truncate text-xs font-medium" title={source.file.name}>{source.file.name}</span>
                      <span className="block truncate text-[11px] text-muted-foreground">{sourceStatus(source, activeId)}</span>
                    </span>
                  </button>
                  <Button variant="ghost" size="icon-xs" disabled={isExtracting} aria-label={`Remove ${source.file.name}`} onClick={() => removeSource(source)} className="mr-1">
                    <X aria-hidden="true" />
                  </Button>
                </li>
              ))}
            </ul>
          </aside>

          <section className="min-h-0 min-w-0 overflow-auto p-4" aria-label="Extraction preview">
            {selected ? (
              <>
                <div className="mb-3 flex flex-wrap items-center gap-2">
                  <h3 className="min-w-0 break-all text-sm font-semibold">{selected.file.name}</h3>
                  <span className="rounded-full bg-muted px-2 py-1 text-[11px]">{sourceStatus(selected, activeId)}</span>
                </div>
                {selected.error && <p role="alert" className="mb-3 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">{selected.error}</p>}
                {result?.reason && <p className="mb-3 rounded-lg border border-border bg-card p-3 text-sm">{result.reason}</p>}
                {result ? (
                  <details className="mb-3 rounded-lg border border-border p-2 text-sm">
                    <summary className="cursor-pointer">View original photo</summary>
                    <Image src={selected.url} width={700} height={500} alt={selected.file.name} unoptimized className="mt-2 max-h-80 w-full object-contain" />
                  </details>
                ) : (
                  <Image src={selected.url} width={700} height={500} alt={selected.file.name} unoptimized className="max-h-80 w-full rounded-lg border border-border bg-muted/30 object-contain" />
                )}
                {result && result.status !== "cannot_convert" && (
                  <>
                    <p className="mb-2 text-xs text-muted-foreground">{result.rows.length} rows. Click a header or cell to edit it. Highlighted cells were unclear in the photo.</p>
                    <div className="overflow-auto rounded-lg border border-border">
                      <table className="w-full text-left text-xs" aria-label="Editable extracted table">
                        <thead className="bg-muted">
                          <tr>{result.columns.map((column, index) => (
                            <th key={index} scope="col" className="p-1">
                              <input aria-label={`Column ${index + 1} name`} maxLength={160} value={column.name} className="w-32 rounded bg-transparent p-1.5 font-semibold focus:bg-background"
                                onChange={(event) => {
                                  const name = event.currentTarget.value
                                  editResult((table) => ({ ...table, columns: table.columns.map((item, i) => i === index ? { ...item, name } : item) }))
                                }} />
                            </th>
                          ))}</tr>
                        </thead>
                        <tbody>{result.rows.map((row, rowIndex) => (
                          <tr key={rowIndex} className="border-t border-border/70">
                            {row.map((value, columnIndex) => {
                              const unclear = unclearCells.has(`${rowIndex}:${columnIndex}`)
                              return (
                                <td key={columnIndex} className={unclear ? "bg-amber-400/20 p-1" : "p-1"}>
                                  <input aria-label={`${result.columns[columnIndex].name || `Column ${columnIndex + 1}`}, row ${rowIndex + 1}${unclear ? ", unclear" : ""}`}
                                    placeholder={unclear ? "Unclear" : undefined} maxLength={32767} value={value ?? ""} className="w-32 rounded bg-transparent p-1.5 focus:bg-background"
                                    onChange={(event) => {
                                      const text = event.currentTarget.value
                                      editResult((table) => ({
                                        ...table,
                                        rows: table.rows.map((cells, index) => index === rowIndex
                                          ? cells.map((cell, col) => col === columnIndex ? text : cell) : cells),
                                        uncertain_cells: table.uncertain_cells.filter((cell) => cell.row !== rowIndex || cell.column !== columnIndex),
                                      }))
                                    }} />
                                </td>
                              )
                            })}
                          </tr>
                        ))}</tbody>
                      </table>
                    </div>
                  </>
                )}
              </>
            ) : (
              <div className="grid h-full min-h-32 place-content-center text-center text-sm text-muted-foreground">
                Add a clear photo of a printed or handwritten log.
              </div>
            )}
          </section>
        </div>

        <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-4 py-3">
          <p role="status" className="text-xs text-muted-foreground">
            {isExtracting ? `Extracting ${sources.find((source) => source.id === activeId)?.file.name ?? "photos"}…`
              : ready.length ? `${ready.length} of ${sources.length} photos included.${unclearCount ? ` ${unclearCount} unclear cells will export blank.` : ""}`
                : "Photos are sent to Gemini when you extract tables."}
          </p>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" disabled={isExtracting || !pending.length} onClick={() => void extractTables()}>
              {isExtracting && <Loader2 aria-hidden="true" className="animate-spin" />}
              {isExtracting ? "Extracting…" : "Extract tables"}
            </Button>
            <Button disabled={isExtracting || !ready.length} onClick={() => downloadWorkbook(sources)}>
              <Download aria-hidden="true" /> Download Excel
            </Button>
          </div>
        </footer>
      </div>
    </dialog>
  )
}
