#!/usr/bin/env python3
"""
MCP Chart Generation Server
Exposes chart generation capabilities as MCP tools
"""

import asyncio
import json
import os
import sys
import platform
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional
import warnings
import logging
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_chart_server.log'),
        logging.StreamHandler(sys.stderr)  # MCP uses stderr for logging
    ]
)
logger = logging.getLogger(__name__)

# MCP imports
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

# Chart generation imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ChartGenerator:
    """Generates pie, bar, and line charts only"""
    
    def __init__(self, output_dir: str = "./charts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('default')
        sns.set_palette("husl")
    
    def open_chart(self, filepath: str):
        """Open chart file automatically based on OS"""
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(filepath)
            elif system == "Darwin":
                subprocess.run(["open", filepath])
            else:
                subprocess.run(["xdg-open", filepath])
        except Exception as e:
            print(f"Could not auto-open chart: {e}")
            print(f"Chart saved at: {filepath}")
    
    def generate_chart(self, chart_type: str, labels: List[str], values: List[float], 
                      title: str = "Chart", description: str = "", auto_open: bool = True) -> Dict[str, Any]:
        """Generate chart based on provided data"""
        try:
            # Validate inputs
            if not labels or not values or len(labels) != len(values):
                return {
                    "success": False,
                    "error": "Labels and values must be provided and have the same length"
                }
            
            # Convert values to float
            try:
                values = [float(v) for v in values]
            except (ValueError, TypeError):
                return {
                    "success": False,
                    "error": "All values must be numeric"
                }
            
            # Validate chart type
            if chart_type not in ["pie", "bar", "line"]:
                return {
                    "success": False,
                    "error": "Chart type must be 'pie', 'bar', or 'line'"
                }
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if chart_type == "pie":
                self._create_pie_chart(labels, values, title, ax)
            elif chart_type == "line":
                self._create_line_chart(labels, values, title, ax)
            else:  # bar
                self._create_bar_chart(labels, values, title, ax)
            
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{chart_type}_chart_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            if auto_open:
                self.open_chart(filepath)
            
            return {
                "success": True,
                "filepath": filepath,
                "filename": filename,
                "chart_type": chart_type,
                "title": title,
                "description": description,
                "data_points": len(labels),
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            plt.close('all')  # Clean up any open figures
            return {
                "success": False,
                "error": f"Error generating chart: {str(e)}"
            }
    
    def _create_pie_chart(self, labels: List[str], values: List[float], title: str, ax):
        """Create pie chart"""
        values = [abs(v) for v in values]  # Ensure positive values for pie chart
        wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title(title, fontsize=16, fontweight='bold')
        return ax
    
    def _create_bar_chart(self, labels: List[str], values: List[float], title: str, ax):
        """Create bar chart"""
        x_pos = range(len(labels))
        bars = ax.bar(x_pos, values, color=sns.color_palette("husl", len(labels)))
        ax.set_xlabel('Categories', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{value:.1f}', 
                   ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
        return ax
    
    def _create_line_chart(self, labels: List[str], values: List[float], title: str, ax):
        """Create line chart"""
        ax.plot(range(len(labels)), values, marker='o', linewidth=3, markersize=8)
        ax.set_xlabel('Categories', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        return ax
    
    def list_charts(self) -> List[Dict[str, Any]]:
        """List all generated charts in the output directory"""
        charts = []
        try:
            if os.path.exists(self.output_dir):
                for filename in os.listdir(self.output_dir):
                    if filename.lower().endswith('.png'):
                        filepath = os.path.join(self.output_dir, filename)
                        stat = os.stat(filepath)
                        charts.append({
                            "filename": filename,
                            "filepath": filepath,
                            "size_bytes": stat.st_size,
                            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
        except Exception as e:
            pass
        
        return sorted(charts, key=lambda x: x['created_at'], reverse=True)

# Initialize chart generator
chart_generator = ChartGenerator()

# Create MCP server
server = Server("chart-generator")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available chart generation tools"""
    return [
        types.Tool(
            name="generate_chart",
            description="Generate pie, bar, or line charts with custom data",
            inputSchema={
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["pie", "bar", "line"],
                        "description": "Type of chart to generate"
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels for data points"
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Numeric values for data points"
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title",
                        "default": "Chart"
                    },
                    "description": {
                        "type": "string",
                        "description": "Chart description",
                        "default": ""
                    },
                    "auto_open": {
                        "type": "boolean",
                        "description": "Automatically open chart after generation",
                        "default": True
                    }
                },
                "required": ["chart_type", "labels", "values"]
            }
        ),
        types.Tool(
            name="list_charts",
            description="List all generated charts in the output directory",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        types.Tool(
            name="get_chart_info",
            description="Get information about the chart generation capabilities",
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    
    if name == "generate_chart":
        # Extract parameters
        chart_type = arguments.get("chart_type")
        labels = arguments.get("labels", [])
        values = arguments.get("values", [])
        title = arguments.get("title", "Chart")
        description = arguments.get("description", "")
        auto_open = arguments.get("auto_open", True)
        
        # Generate chart
        result = chart_generator.generate_chart(
            chart_type=chart_type,
            labels=labels,
            values=values,
            title=title,
            description=description,
            auto_open=auto_open
        )
        
        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "list_charts":
        charts = chart_generator.list_charts()
        
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "charts": charts,
                "total_count": len(charts),
                "output_directory": chart_generator.output_dir
            }, indent=2)
        )]
    
    elif name == "get_chart_info":
        info = {
            "server_name": "Chart Generation MCP Server",
            "supported_chart_types": ["pie", "bar", "line"],
            "output_directory": chart_generator.output_dir,
            "output_format": "PNG",
            "features": [
                "Automatic chart opening",
                "Custom titles and descriptions",
                "High-resolution output (300 DPI)",
                "Professional styling with seaborn",
                "Support for multiple data visualization types"
            ],
            "chart_capabilities": {
                "pie": "Shows proportional data with percentages",
                "bar": "Compares categorical data with value labels",
                "line": "Shows trends and changes over categories"
            }
        }
        
        return [types.TextContent(
            type="text",
            text=json.dumps(info, indent=2)
        )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main function to run the MCP server"""
    logger.info("Starting MCP Chart Generation Server...")
    logger.info(f"Server capabilities: Tools (listChanged=True)")
    
    try:
        # Server transport using stdio
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("STDIO server started successfully")
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="chart-generator",
                    server_version="1.0.0",
                    capabilities=types.ServerCapabilities(
                        tools=types.ToolsCapability(listChanged=True)
                    )
                )
            )
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

def test_chart_generation():
    """Test function to verify chart generation works"""
    print("Testing chart generation...")
    
    # Test pie chart
    result = chart_generator.generate_chart(
        chart_type="pie",
        labels=["Python", "JavaScript", "Java", "C++"],
        values=[40, 30, 20, 10],
        title="Programming Languages Usage",
        auto_open=False
    )
    
    print(f"Pie chart test result: {result['success']}")
    if result['success']:
        print(f"Chart saved to: {result['filepath']}")
    else:
        print(f"Error: {result['error']}")
    
    # Test bar chart
    result = chart_generator.generate_chart(
        chart_type="bar",
        labels=["Q1", "Q2", "Q3", "Q4"],
        values=[100, 150, 120, 180],
        title="Quarterly Sales",
        auto_open=False
    )
    
    print(f"Bar chart test result: {result['success']}")
    if result['success']:
        print(f"Chart saved to: {result['filepath']}")
    else:
        print(f"Error: {result['error']}")
    
    # List existing charts
    charts = chart_generator.list_charts()
    print(f"Total charts generated: {len(charts)}")
    
    return True

if __name__ == "__main__":
    # Check if we should run in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Running in test mode...")
        test_chart_generation()
        print("Test completed!")
    else:
        print("Starting MCP Chart Server (use 'python mcp_chart_server.py test' to test chart generation)")
        asyncio.run(main())