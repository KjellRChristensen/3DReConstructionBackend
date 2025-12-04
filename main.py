"""
3D Reconstruction Pipeline - Main Entry Point
"""
import argparse
import sys
import socket
from pathlib import Path

import uvicorn


def run_server(host: str = "0.0.0.0", port: int = 7001, reload: bool = False):
    """Run the API server"""
    # Get local IP address
    local_ip = "localhost"
    if host == "0.0.0.0":
        try:
            # Get actual local IP for display
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "127.0.0.1"
    else:
        local_ip = host

    print("\n" + "="*70)
    print("  3D Reconstruction Backend Server")
    print("="*70)
    print(f"  🚀 Server starting...")
    print(f"  📡 Host: {host}")
    print(f"  🔌 Port: {port}")
    print(f"  🌐 Local URL: http://{local_ip}:{port}")
    print(f"  📚 API Docs: http://{local_ip}:{port}/docs")
    print(f"  🔄 Auto-reload: {'enabled' if reload else 'disabled'}")
    print("="*70 + "\n")

    uvicorn.run(
        "src.api.server:app",
        host=host,
        
        
        port=port,
        reload=reload,
    )


def run_cli(input_path: str, output_dir: str, formats: str, **kwargs):
    """Run pipeline from command line"""
    from src.pipeline import run_pipeline

    export_formats = [f.strip() for f in formats.split(",")]

    result = run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        export_formats=export_formats,
        **kwargs
    )

    if result.success:
        print(f"Success! Output files:")
        for f in result.output_files:
            print(f"  - {f}")
    else:
        print(f"Failed: {result.error}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="3D Reconstruction Pipeline - Convert 2D floor plans to 3D models"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Server command
    server_parser = subparsers.add_parser("server", help="Run API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=7001, help="Port to listen on")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process a floor plan")
    process_parser.add_argument("input", help="Input file (PDF, PNG, JPG, DXF)")
    process_parser.add_argument("-o", "--output", default="./output", help="Output directory")
    process_parser.add_argument(
        "-f", "--formats",
        default="glb,obj",
        help="Export formats (comma-separated: obj,glb,gltf,stl,usdz,ifc)"
    )
    process_parser.add_argument("--wall-height", type=float, help="Wall height in meters")
    process_parser.add_argument("--floors", type=int, default=1, help="Number of floors")
    process_parser.add_argument("--dpi", type=int, default=300, help="Processing DPI")

    args = parser.parse_args()

    if args.command == "server":
        run_server(host=args.host, port=args.port, reload=args.reload)
    elif args.command == "process":
        kwargs = {}
        if args.wall_height:
            kwargs["wall_height"] = args.wall_height
        if args.floors:
            kwargs["num_floors"] = args.floors
        if args.dpi:
            kwargs["dpi"] = args.dpi

        run_cli(args.input, args.output, args.formats, **kwargs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
