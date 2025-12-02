# Frontend API Reference

REST API endpoints for the iOS Swift frontend to communicate with the 3D Reconstruction backend.

**Base URL:** `http://localhost:7001`

---

## System Endpoints

### Check Health
```
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### Check GPU Status
```
GET /system/gpu
```
**Response:**
```json
{
  "pytorch_installed": true,
  "device": "mps",
  "mps_available": true,
  "mps_built": true,
  "cuda_available": false,
  "device_name": "Apple Silicon GPU (MPS)",
  "pytorch_version": "2.9.1"
}
```

---

## File Management

### List Input Files
```
GET /files/input
```
**Response:**
```json
{
  "files": [
    {
      "name": "sample_floorplan.png",
      "path": "data/input/sample_floorplan.png",
      "size": 30720,
      "size_human": "30.0 KB",
      "modified": "2024-01-15T10:30:00",
      "type": "image",
      "extension": ".png"
    }
  ],
  "total": 1,
  "directory": "/path/to/data/input"
}
```

### Upload Input File
```
POST /files/input
Content-Type: multipart/form-data

file: <binary>
```
**Response:**
```json
{
  "status": "uploaded",
  "file": {
    "name": "my_floorplan.png",
    "path": "data/input/my_floorplan.png",
    "size": 45000,
    "size_human": "43.9 KB",
    "type": "image"
  }
}
```

### Get Input File (Preview/Download)
```
GET /files/input/{filename}
```
Returns file binary with appropriate Content-Type header.

### Delete Input File
```
DELETE /files/input/{filename}
```
**Response:**
```json
{
  "status": "deleted",
  "filename": "my_floorplan.png"
}
```

### List Output Files
```
GET /files/output
```
Same response format as `/files/input`.

### Get Output File (Download 3D Model)
```
GET /files/output/{filepath}
```
Returns file binary (GLB, OBJ, etc.).

---

## Reconstruction Strategies

### List Available Strategies
```
GET /strategies
```
**Response:**
```json
{
  "strategies": [
    {
      "id": "external_api",
      "name": "External DNN API",
      "description": "Use external AI services (Kaedim, Meshy, Replicate, Tripo3D)",
      "services": ["kaedim", "meshy", "replicate", "tripo3d"],
      "requires_api_key": true,
      "best_for": "High-quality results, production use",
      "accuracy": "85-95%",
      "speed": "1-5 minutes",
      "available": false
    },
    {
      "id": "basic_extrusion",
      "name": "Basic Extrusion",
      "description": "Built-in wall detection and extrusion",
      "services": [],
      "requires_api_key": false,
      "best_for": "Quick previews, simple floor plans",
      "accuracy": "60-80%",
      "speed": "Seconds",
      "available": true
    },
    {
      "id": "multi_view_dnn",
      "name": "Multi-View DNN",
      "description": "Deep learning reconstruction (depth estimation, GaussianCAD)",
      "services": ["depth_estimation", "gaussian_cad", "cad2program"],
      "requires_api_key": false,
      "best_for": "Multiple views, complex buildings",
      "accuracy": "80-95%",
      "speed": "30 seconds - 2 minutes",
      "available": true
    }
  ],
  "default": "basic_extrusion",
  "recommended": "multi_view_dnn"
}
```

---

## Reconstruction Jobs

### Quick Preview (Synchronous)
```
POST /reconstruct/preview?filename={filename}&wall_height={height}
```
**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| filename | string | required | File name in input folder |
| wall_height | float | 2.8 | Wall height in meters |

**Response:**
```json
{
  "success": true,
  "output_file": "previews/sample_floorplan_preview.glb",
  "download_url": "/files/output/previews/sample_floorplan_preview.glb",
  "format": "glb",
  "strategy": "Basic Extrusion (Built-in)"
}
```

### Full Reconstruction (Async)
```
POST /reconstruct
```
**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| filename | string | required | File name in input folder |
| strategy | enum | auto | `external_api`, `basic_extrusion`, `multi_view_dnn`, `auto` |
| service | string | null | For external_api: `kaedim`, `meshy`, `replicate`, `tripo3d` |
| model_type | string | depth_estimation | For multi_view_dnn: `depth_estimation`, `gaussian_cad` |
| wall_height | float | 2.8 | Wall height in meters |
| export_format | string | glb | Output format: `glb`, `obj`, `stl` |
| additional_views | string | null | Comma-separated filenames for multi-view |

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "strategy": "multi_view_dnn",
  "message": "Reconstruction started with multi_view_dnn strategy"
}
```

### Get Job Status
```
GET /jobs/{job_id}
```
**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 0.65,
  "current_stage": "reconstruction",
  "created_at": "2024-01-15T10:30:00",
  "completed_at": null,
  "error": null,
  "output_files": []
}
```

**Status Values:**
- `pending` - Job queued
- `processing` - Actively processing
- `completed` - Successfully finished
- `failed` - Error occurred

**Stage Values:**
- `initializing`
- `ingestion`
- `vectorization`
- `recognition`
- `reconstruction`
- `saving_output`
- `complete`
- `failed`

### Get Job Progress (Detailed)
```
GET /jobs/{job_id}/progress
```
**Response:**
```json
{
  "stage": "reconstruction",
  "progress": 0.65,
  "message": "Processing: reconstruction"
}
```

### List All Jobs
```
GET /jobs?limit=50
```
**Response:** Array of JobResponse objects.

### Download Job Output
```
GET /jobs/{job_id}/download/{filename}
```
Returns binary file.

### Delete Job
```
DELETE /jobs/{job_id}
```
Removes job and associated files.

---

## Validation Endpoints

Validate reconstruction quality against ground truth 3D CAD models.

### List Supported Formats
```
GET /validation/formats
```
**Response:**
```json
{
  "supported_formats": [
    {"extension": ".ifc", "name": "IFC (BIM)", "description": "Industry Foundation Classes"},
    {"extension": ".obj", "name": "OBJ", "description": "Wavefront 3D mesh"},
    {"extension": ".glb", "name": "GLB", "description": "GL Transmission Format (binary)"},
    {"extension": ".stl", "name": "STL", "description": "Stereolithography mesh"},
    {"extension": ".dxf", "name": "DXF", "description": "AutoCAD Drawing Exchange Format"}
  ],
  "metrics_computed": [
    {"name": "chamfer_distance", "description": "Average nearest-neighbor distance (lower is better)"},
    {"name": "hausdorff_distance", "description": "Maximum deviation (lower is better)"},
    {"name": "iou_3d", "description": "3D Intersection over Union (higher is better, 0-1)"},
    {"name": "f_score", "description": "F-score at distance threshold (higher is better, 0-1)"}
  ]
}
```

### Run Validation Pipeline
```
POST /validation/run?ground_truth_file={filename}&strategy={strategy}&wall_height={height}&floor_height={height}
```
Runs full validation: Load CAD → Generate 2D → Reconstruct 3D → Compare to ground truth.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "pending",
  "message": "Validation started for building.ifc"
}
```

### Compare Two Meshes Directly
```
POST /validation/compare?predicted_file={pred}&ground_truth_file={gt}
```
**Response:**
```json
{
  "success": true,
  "metrics": {
    "chamfer_distance": 0.05,
    "hausdorff_distance": 0.15,
    "iou_3d": 0.85,
    "f_score": 0.92,
    "precision": 0.94,
    "recall": 0.90
  },
  "summary": "Comparison Results:\n  Chamfer Distance: 0.0500\n  ..."
}
```

### Generate 2D Projection from 3D Model
```
POST /validation/project?model_file={filename}&floor_height={height}&resolution={res}
```
**Response:**
```json
{
  "success": true,
  "output_file": "projections/model_projection.png",
  "download_url": "/files/output/projections/model_projection.png",
  "projection_info": {
    "width": 1024,
    "height": 768,
    "scale": 0.01,
    "floor_height": 1.0
  }
}
```

### Generate Training Data Pair
```
POST /validation/training-pair?model_file={filename}&floor_height={height}
```
Creates paired training data: 2D floor plan (X) + 3D model (Y) + metadata.

**Response:**
```json
{
  "success": true,
  "files": {
    "x_2d": "training_pairs/model_2d.png",
    "y_3d": "training_pairs/model_3d.glb",
    "metadata": "training_pairs/model_meta.json"
  },
  "download_urls": {
    "x_2d": "/files/output/training_pairs/model_2d.png",
    "y_3d": "/files/output/training_pairs/model_3d.glb",
    "metadata": "/files/output/training_pairs/model_meta.json"
  }
}
```

---

## iOS Swift Integration Examples

### SwiftUI File Upload
```swift
func uploadFloorPlan(image: UIImage) async throws -> String {
    guard let imageData = image.pngData() else {
        throw AppError.invalidImage
    }

    var request = URLRequest(url: URL(string: "\(baseURL)/files/input")!)
    request.httpMethod = "POST"

    let boundary = UUID().uuidString
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

    var body = Data()
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"file\"; filename=\"floorplan.png\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
    body.append(imageData)
    body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

    request.httpBody = body

    let (data, _) = try await URLSession.shared.data(for: request)
    let response = try JSONDecoder().decode(UploadResponse.self, from: data)
    return response.file.name
}
```

### Start Reconstruction
```swift
func startReconstruction(filename: String, strategy: String = "auto") async throws -> String {
    var components = URLComponents(string: "\(baseURL)/reconstruct")!
    components.queryItems = [
        URLQueryItem(name: "filename", value: filename),
        URLQueryItem(name: "strategy", value: strategy),
        URLQueryItem(name: "wall_height", value: "2.8")
    ]

    var request = URLRequest(url: components.url!)
    request.httpMethod = "POST"

    let (data, _) = try await URLSession.shared.data(for: request)
    let response = try JSONDecoder().decode(ReconstructResponse.self, from: data)
    return response.job_id
}
```

### Poll Job Status
```swift
func pollJobStatus(jobId: String) -> AsyncStream<JobStatus> {
    AsyncStream { continuation in
        Task {
            while true {
                let url = URL(string: "\(baseURL)/jobs/\(jobId)")!
                let (data, _) = try await URLSession.shared.data(from: url)
                let status = try JSONDecoder().decode(JobStatus.self, from: data)

                continuation.yield(status)

                if status.status == "completed" || status.status == "failed" {
                    continuation.finish()
                    break
                }

                try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
            }
        }
    }
}
```

### Download 3D Model for AR Quick Look
```swift
func downloadModel(jobId: String, filename: String) async throws -> URL {
    let url = URL(string: "\(baseURL)/jobs/\(jobId)/download/\(filename)")!
    let (tempURL, _) = try await URLSession.shared.download(from: url)

    // Move to permanent location
    let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let destinationURL = documentsURL.appendingPathComponent(filename)

    try FileManager.default.moveItem(at: tempURL, to: destinationURL)
    return destinationURL
}
```

---

## Model Definitions (Swift)

```swift
struct UploadResponse: Codable {
    let status: String
    let file: FileInfo
}

struct FileInfo: Codable {
    let name: String
    let path: String
    let size: Int
    let sizeHuman: String
    let type: String

    enum CodingKeys: String, CodingKey {
        case name, path, size, type
        case sizeHuman = "size_human"
    }
}

struct ReconstructResponse: Codable {
    let jobId: String
    let status: String
    let strategy: String
    let message: String

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case status, strategy, message
    }
}

struct JobStatus: Codable {
    let jobId: String
    let status: String
    let progress: Double
    let currentStage: String?
    let createdAt: String
    let completedAt: String?
    let error: String?
    let outputFiles: [String]

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case status, progress
        case currentStage = "current_stage"
        case createdAt = "created_at"
        case completedAt = "completed_at"
        case error
        case outputFiles = "output_files"
    }
}

struct Strategy: Codable {
    let id: String
    let name: String
    let description: String
    let services: [String]
    let requiresApiKey: Bool
    let bestFor: String
    let accuracy: String
    let speed: String
    let available: Bool

    enum CodingKeys: String, CodingKey {
        case id, name, description, services, accuracy, speed, available
        case requiresApiKey = "requires_api_key"
        case bestFor = "best_for"
    }
}

struct GPUStatus: Codable {
    let pytorchInstalled: Bool
    let device: String
    let mpsAvailable: Bool
    let mpsBuilt: Bool
    let cudaAvailable: Bool
    let deviceName: String
    let pytorchVersion: String?

    enum CodingKeys: String, CodingKey {
        case pytorchInstalled = "pytorch_installed"
        case device
        case mpsAvailable = "mps_available"
        case mpsBuilt = "mps_built"
        case cudaAvailable = "cuda_available"
        case deviceName = "device_name"
        case pytorchVersion = "pytorch_version"
    }
}
```

---

## Error Handling

All endpoints return errors in the format:
```json
{
  "detail": "Error description"
}
```

**HTTP Status Codes:**
- `200` - Success
- `201` - Created
- `400` - Bad request (invalid parameters)
- `404` - Resource not found
- `500` - Server error
- `503` - Service unavailable (dependencies missing)

---

## Frontend UI Recommendations

### Strategy Selection UI
Display strategies as cards with:
- Name and description
- Availability indicator (green/red dot)
- Speed and accuracy badges
- "Requires API key" warning if applicable

### Progress Indicator
- Use `progress` (0.0-1.0) for progress bar
- Display `current_stage` as status text
- Poll every 1-2 seconds during processing

### Output Formats
- `.glb` - Best for web viewers and Three.js
- `.usdz` - Use for iOS AR Quick Look
- `.obj` - For compatibility with other 3D software

### Recommended Flow
1. User uploads floor plan image
2. Show preview with quick `/reconstruct/preview`
3. User adjusts parameters (wall height, strategy)
4. Start full reconstruction with `/reconstruct`
5. Poll status and show progress
6. Download and display 3D model
7. Option to export in different formats
