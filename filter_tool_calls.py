#!/usr/bin/env python3
"""
Filter tool calls export by status (success/failure)
Ultra-fast version using line-by-line JSON parsing
"""

import json
import sys
from pathlib import Path

def filter_tool_calls(input_file, output_dir=None):
    """
    Filter tool calls by status - handles multi-line JSON objects
    Optimized for speed with large files
    
    Args:
        input_file: Path to the combined tool calls export file
        output_dir: Directory to save filtered files
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Input file not found: {input_file}")
        return
    
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    success_file = output_dir / f"{input_path.stem}_successful.txt"
    failed_file = output_dir / f"{input_path.stem}_failed.txt"
    
    success_count = 0
    failed_count = 0
    error_count = 0
    
    print(f"Reading: {input_file}")
    print(f"Processing multi-line JSON objects...")
    print(f"Real-time streaming to output files\n")
    
    try:
        with open(input_path, 'r', buffering=1024*1024) as infile, \
             open(success_file, 'w', buffering=1024*64) as success_out, \
             open(failed_file, 'w', buffering=1024*64) as failed_out:
            
            last_flush = 0
            buffer = ""
            
            for line in infile:
                buffer += line
                
                if line.strip() == '}':
                    try:
                        obj = json.loads(buffer)
                        status = obj.get('status', 'unknown')
                        
                        json_line = json.dumps(obj)
                        
                        if status == 'success':
                            success_out.write(json_line + '\n')
                            success_count += 1
                        else:
                            failed_out.write(json_line + '\n')
                            failed_count += 1
                        
                        total = success_count + failed_count
                        
                        if total - last_flush >= 5000:
                            success_out.flush()
                            failed_out.flush()
                            print(f"  {total:,} calls processed ({success_count:,} success | {failed_count:,} failed)")
                            last_flush = total
                        
                        buffer = ""
                    
                    except json.JSONDecodeError as e:
                        error_count += 1
                        if error_count <= 3:
                            print(f"  Invalid JSON: {str(e)[:60]}")
                        buffer = ""
                    except Exception as e:
                        error_count += 1
                        if error_count <= 3:
                            print(f"  Error: {str(e)[:60]}")
                        buffer = ""
            
            success_out.flush()
            failed_out.flush()
        
        print("\n" + "="*70)
        print("FILTERING COMPLETE")
        print("="*70)
        print(f"Successful calls: {success_count:,}")
        print(f"Failed calls: {failed_count:,}")
        print(f"Processing errors: {error_count}")
        print(f"Total processed: {success_count + failed_count:,}")
        print("="*70)
        
        try:
            success_size = success_file.stat().st_size / (1024*1024*1024)
            failed_size = failed_file.stat().st_size / (1024*1024*1024)
            print(f"Success file: {success_file}")
            print(f"   Size: {success_size:.2f} GB ({success_file.stat().st_size:,} bytes)")
            print(f"\nFailed file: {failed_file}")
            print(f"   Size: {failed_size:.2f} GB ({failed_file.stat().st_size:,} bytes)")
        except:
            pass
        
        print("="*70)
        print("All files successfully created and saved to disk!")
        print("="*70)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_tool_calls.py <input_file> [output_directory]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    filter_tool_calls(input_file, output_dir)