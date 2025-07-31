#!/usr/bin/env python3

"""
PDF RAG Conversational Agent
An intelligent agent that uses the MCP PDF RAG server for document analysis and chat
"""

import asyncio
import json
import subprocess
import sys
import os
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
import signal

class RAGAgent:
    """Conversational agent using MCP PDF RAG server"""
    
    def __init__(self):
        self.server_process = None
        self.request_id = 0
        self.conversation_history = []
        self.loaded_documents = []
        self.current_session = {
            "started_at": datetime.now().isoformat(),
            "queries_count": 0,
            "charts_generated": 0
        }
    
    def get_next_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id
    
    async def start_rag_server(self):
        """Start the MCP RAG server"""
        print("Starting Enhanced PDF RAG Server...")
        try:
            self.server_process = subprocess.Popen(
                [sys.executable, "mcp_rag_server.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Give server time to initialize
            await asyncio.sleep(4)
            print("RAG Server started successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to start RAG server: {e}")
            return False
    
    async def send_mcp_request(self, method: str, params: Optional[Dict] = None, timeout: float = 30.0) -> Dict[str, Any]:
        """Send request to MCP server"""
        request = {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": method
        }
        
        if params:
            request["params"] = params
        
        try:
            # Check if server is still running
            if self.server_process.poll() is not None:
                return {"error": "Server process has died"}
            
            request_json = json.dumps(request, separators=(',', ':')) + "\n"
            self.server_process.stdin.write(request_json)
            self.server_process.stdin.flush()
            
            await asyncio.sleep(0.3)  # Give server time to process
            
            response_line = await asyncio.wait_for(
                asyncio.to_thread(self.server_process.stdout.readline),
                timeout=timeout
            )
            
            if not response_line or not response_line.strip():
                if self.server_process.poll() is not None:
                    return {"error": "Server process terminated"}
                return {"error": "No response from server"}
            
            try:
                response = json.loads(response_line.strip())
                return response
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON response: {e}", "raw_response": response_line}
            
        except asyncio.TimeoutError:
            return {"error": f"Request timed out after {timeout} seconds"}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {e}"}
        except Exception as e:
            return {"error": f"Communication error: {e}"}
    
    async def send_notification(self, method: str, params: Optional[Dict] = None):
        """Send notification to MCP server"""
        notification = {
            "jsonrpc": "2.0",
            "method": method
        }
        
        if params:
            notification["params"] = params
        
        try:
            notification_json = json.dumps(notification, separators=(',', ':')) + "\n"
            self.server_process.stdin.write(notification_json)
            self.server_process.stdin.flush()
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Failed to send notification: {e}")
    
    async def initialize_mcp_connection(self) -> bool:
        """Initialize MCP connection"""
        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "pdf-rag-agent",
                "version": "1.0.0"
            }
        }
        
        response = await self.send_mcp_request("initialize", init_params)
        
        if "error" in response:
            print(f"MCP initialization failed: {response['error']}")
            return False
        
        if "result" in response:
            # Send initialized notification
            await self.send_notification("notifications/initialized")
            print("MCP connection established!")
            return True
        
        return False
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: float = None) -> Dict[str, Any]:
        """Call MCP tool and return parsed result"""
        params = {
            "name": tool_name,
            "arguments": arguments
        }
        
        # Set appropriate timeout based on tool type
        if timeout is None:
            if tool_name in ["generate_chart", "query_documents"]:
                timeout = 60.0  # Longer timeout for heavy operations
            elif tool_name == "load_pdf":
                timeout = 45.0  # Medium timeout for PDF loading
            else:
                timeout = 30.0  # Default timeout
        
        response = await self.send_mcp_request("tools/call", params, timeout)
        
        if "error" in response:
            return {"success": False, "error": response["error"]}
        
        result = response.get("result", {})
        content = result.get("content", [])
        
        if content and len(content) > 0:
            text_content = content[0].get("text", "")
            try:
                return json.loads(text_content)
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"Invalid JSON response: {e}", "raw_response": text_content}
        
        return {"success": False, "error": "No content in response"}
    
    async def load_document(self, pdf_path: str) -> bool:
        """Load a PDF document"""
        print(f"Loading document: {pdf_path}")
        
        result = await self.call_mcp_tool("load_pdf", {
            "pdf_path": pdf_path,
            "force_reload": False
        })
        
        if result.get("success"):
            doc_name = os.path.basename(pdf_path)
            if doc_name not in self.loaded_documents:
                self.loaded_documents.append(doc_name)
            
            chunks = result.get("chunks_count", 0)
            text_length = result.get("text_length", 0)
            print(f"Document loaded successfully!")
            print(f"   {doc_name}")
            print(f"   {chunks} text chunks created")
            print(f"   {text_length:,} characters extracted")
            return True
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"Failed to load document: {error_msg}")
            
            # Provide helpful tips based on error type
            if "not found" in error_msg.lower():
                print(f"   Tip: Check if the file path is correct and the file exists")
            elif "no text could be extracted" in error_msg.lower():
                print(f"   Tip: This might be an image-based PDF. Try using OCR software first")
            elif "not a pdf" in error_msg.lower():
                print(f"   Tip: Make sure the file has a .pdf extension and is a valid PDF")
            elif "permission" in error_msg.lower():
                print(f"   Tip: Check if the file is open in another program or if you have read permissions")
            
            return False
    
    async def query_documents(self, question: str) -> Dict[str, Any]:
        """Query the loaded documents"""
        self.current_session["queries_count"] += 1
        
        result = await self.call_mcp_tool("query_documents", {
            "question": question,
            "top_k": 5,
            "generate_chart": True
        })
        
        if result.get("success"):
            if result.get("found_results"):
                # Track chart generation
                if result.get("chart_generated"):
                    self.current_session["charts_generated"] += 1
                
                # Add to conversation history
                self.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "answer": result.get("llm_answer", ""),
                    "confidence": result.get("confidence", 0),
                    "chart_generated": result.get("chart_generated") is not None
                })
        
        return result
    
    async def generate_custom_chart(self, chart_type: str, labels: List[str], values: List[float], title: str = "Chart") -> Dict[str, Any]:
        """Generate a custom chart"""
        result = await self.call_mcp_tool("generate_chart", {
            "chart_type": chart_type,
            "labels": labels,
            "values": values,
            "title": title,
            "auto_open": False  # Don't auto-open to avoid issues
        })
        
        if result.get("success"):
            self.current_session["charts_generated"] += 1
        
        return result
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return await self.call_mcp_tool("get_system_stats", {})
    
    async def list_loaded_documents(self) -> Dict[str, Any]:
        """List currently loaded documents"""
        return await self.call_mcp_tool("list_loaded_documents", {})
    
    async def reset_system(self) -> Dict[str, Any]:
        """Reset the RAG system"""
        result = await self.call_mcp_tool("reset_system", {})
        if result.get("success"):
            self.loaded_documents = []
            self.conversation_history = []
            self.current_session = {
                "started_at": datetime.now().isoformat(),
                "queries_count": 0,
                "charts_generated": 0
            }
        return result
    
    def parse_user_command(self, user_input: str) -> Dict[str, Any]:
        """Parse user input to determine intent and extract parameters"""
        user_input = user_input.strip()
        
        # Load document commands
        if user_input.lower().startswith(('load ', 'open ', 'add ')):
            path_match = re.search(r'(?:load|open|add)\s+(.+)', user_input, re.IGNORECASE)
            if path_match and path_match.group(1):
                path = path_match.group(1).strip() if path_match.group(1) else ""
                if path:
                    return {"action": "load", "path": path}
        
        # Chart generation commands
        chart_match = re.search(r'(?:create|generate|make)\s+(?:a\s+)?(\w+)\s+chart.*?(?:with|using|for)?\s*(?:labels?)?\s*(?:\[([^\]]+)\])?\s*(?:(?:and|with)\s*)?(?:values?)?\s*(?:\[([^\]]+)\])?', user_input, re.IGNORECASE)
        if chart_match:
            chart_type = chart_match.group(1).lower()
            labels_str = chart_match.group(2)
            values_str = chart_match.group(3)
            
            if chart_type in ['pie', 'bar', 'line']:
                labels = []
                values = []
                
                if labels_str:
                    labels = [l.strip().strip('"\'') for l in labels_str.split(',') if l and l.strip()]
                if values_str:
                    try:
                        values = [float(v.strip()) for v in values_str.split(',') if v and v.strip()]
                    except ValueError:
                        pass
                
                return {
                    "action": "chart",
                    "chart_type": chart_type,
                    "labels": labels,
                    "values": values
                }
        
        # System commands
        if user_input.lower() in ['status', 'stats', 'system status', 'show status']:
            return {"action": "status"}
        
        if user_input.lower() in ['list documents', 'list docs', 'show documents', 'documents']:
            return {"action": "list_docs"}
        
        if user_input.lower() in ['list charts', 'show charts', 'charts']:
            return {"action": "list_charts"}
        
        if user_input.lower() in ['reset', 'clear', 'reset system']:
            return {"action": "reset"}
        
        if user_input.lower() in ['help', '?', 'commands']:
            return {"action": "help"}
        
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            return {"action": "quit"}
        
        # Default to query
        return {"action": "query", "question": user_input}
    
    def display_help(self):
        """Display help information"""
        print("\n" + "="*70)
        print("PDF RAG AGENT - HELP")
        print("="*70)
        print("DOCUMENT MANAGEMENT:")
        print("  load <path>              - Load a PDF document")
        print("  documents                - List loaded documents")
        print("  charts                   - List generated charts")
        print("  reset                    - Clear all data and start fresh")
        print()
        print("QUERYING:")
        print("  <your question>          - Ask anything about your documents")
        print("  Examples:")
        print("    • What is this document about?")
        print("    • Summarize the key findings")
        print("    • Show me the statistics")
        print("    • What are the main conclusions?")
        print()
        print("CHART GENERATION:")
        print("  create <type> chart      - Generate charts with data from questions")
        print("  Chart types: pie, bar, line")
        print("  Example: create pie chart with labels [A,B,C] values [10,20,30]")
        print()
        print("SYSTEM:")
        print("  status                   - Show system information")
        print("  help                     - Show this help")
        print("  quit                     - Exit the agent")
        print("="*70)
    
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "="*70)
        print("    WELCOME TO THE PDF RAG CONVERSATIONAL AGENT")
        print("="*70)
        print()
        print("I'm your intelligent document assistant! Here's what I can do:")
        print()
        print("LOAD & ANALYZE PDFs")
        print("   - Load any PDF document for intelligent analysis")
        print("   - Extract and chunk text for optimal retrieval")
        print("   - Build semantic search indexes")
        print()
        print("SMART QUERYING")
        print("   - Answer questions using advanced AI (Mistral LLM)")
        print("   - Find relevant information across your documents")
        print("   - Provide confidence scores for answers")
        print()
        print("AUTO CHART GENERATION")
        print("   - Automatically create charts from numeric data")
        print("   - Support for pie, bar, and line charts")
        print("   - Charts open automatically for immediate viewing")
        print()
        print("NATURAL CONVERSATION")
        print("   - Chat naturally about your documents")
        print("   - Ask follow-up questions")
        print("   - Get summaries, analysis, and insights")
        print()
        print("Type 'help' for commands or just start asking questions!")
        print("="*70)
    
    def display_session_summary(self):
        """Display session summary"""
        print("\n" + "="*62)
        print("SESSION SUMMARY")
        print("="*62)
        print(f"Started: {self.current_session['started_at']}")
        print(f"Documents loaded: {len(self.loaded_documents)}")
        print(f"Questions asked: {self.current_session['queries_count']}")
        print(f"Charts generated: {self.current_session['charts_generated']}")
        
        if self.conversation_history:
            print(f"Last conversation topics:")
            for i, conv in enumerate(self.conversation_history[-3:], 1):
                question = conv['question'][:50] + "..." if len(conv['question']) > 50 else conv['question']
                print(f"   {i}. {question}")
        
        print("="*62)
    
    async def interactive_chat(self):
        """Main interactive chat loop"""
        self.display_welcome()
        
        # Check for PDFs in current directory
        pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
        if pdf_files:
            print(f"\nFound PDF files in current directory:")
            for i, pdf in enumerate(pdf_files[:5], 1):
                print(f"   {i}. {pdf}")
            
            print(f"\nTip: Type 'load <filename>' to load a document")
        
        print(f"\n{'Agent':<12}: Ready! What would you like to do?")
        print("-" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n{'You':<12}: ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                command = self.parse_user_command(user_input)
                action = command.get("action")
                

                
                if action == "quit":
                    print(f"\n{'Agent':<12}: Thank you for using the PDF RAG Agent!")
                    self.display_session_summary()
                    break
                
                elif action == "help":
                    self.display_help()
                
                elif action == "load":
                    path = command.get("path", "")
                    if path:
                        await self.load_document(path)
                    else:
                        print(f"{'Agent':<12}: Please specify a file path to load")
                
                elif action == "status":
                    print(f"\n{'Agent':<12}: Getting system status...")
                    status = await self.get_system_status()
                    if "llm_model" in status:
                        print(f"{'Agent':<12}: System Status:")
                        print(f"{'':>14}LLM: {status['llm_model']}")
                        print(f"{'':>14}Chunks: {status['chromadb_chunks']}")
                        print(f"{'':>14}Documents: {status['loaded_documents']}")
                        print(f"{'':>14}Charts: {len(status.get('loaded_document_details', {}))}")
                    else:
                        print(f"{'Agent':<12}: Could not get system status")
                
                elif action == "list_docs":
                    docs = await self.list_loaded_documents()
                    if docs.get("total_count", 0) > 0:
                        print(f"{'Agent':<12}: Loaded Documents:")
                        for doc_name, info in docs.get("loaded_documents", {}).items():
                            print(f"{'':>14}- {doc_name} ({info.get('chunks_count', 0)} chunks)")
                    else:
                        print(f"{'Agent':<12}: No documents loaded yet")
                
                elif action == "list_charts":
                    charts_result = await self.call_mcp_tool("list_charts", {})
                    if charts_result.get("success", True) and "charts" in charts_result:
                        charts = charts_result.get("charts", [])
                        total = charts_result.get("total_count", len(charts))
                        if total > 0:
                            print(f"{'Agent':<12}: Generated Charts ({total} total):")
                            for i, chart in enumerate(charts[:10], 1):  # Show first 10
                                filename = chart.get("filename", "Unknown")
                                size_kb = chart.get("size_bytes", 0) // 1024
                                created = chart.get("created_at", "Unknown")[:16]  # Just date and time
                                print(f"{'':>14}{i}. {filename} ({size_kb}KB) - {created}")
                            
                            if total > 10:
                                print(f"{'':>14}... and {total - 10} more charts")
                        else:
                            print(f"{'Agent':<12}: No charts generated yet")
                    else:
                        error_msg = charts_result.get("error", "Could not retrieve charts")
                        print(f"{'Agent':<12}: Failed to list charts: {error_msg}")
                
                elif action == "reset":
                    print(f"{'Agent':<12}: Are you sure you want to reset everything? (y/N)")
                    confirm = input(f"{'You':<12}: ").strip().lower()
                    if confirm in ['y', 'yes']:
                        result = await self.reset_system()
                        if result.get("success"):
                            print(f"{'Agent':<12}: System reset successfully!")
                        else:
                            print(f"{'Agent':<12}: Reset failed: {result.get('error', 'Unknown error')}")
                    else:
                        print(f"{'Agent':<12}: Reset cancelled.")
                
                elif action == "chart":
                    chart_type = command.get("chart_type")
                    labels = command.get("labels", [])
                    values = command.get("values", [])
                    
                    if labels and values and len(labels) == len(values):
                        print(f"{'Agent':<12}: Creating {chart_type} chart...")
                        result = await self.generate_custom_chart(chart_type, labels, values, f"{chart_type.title()} Chart")
                        if result.get("success"):
                            filename = result.get('filename', 'chart')
                            chart_path = result.get('filepath')
                            print(f"{'Agent':<12}: Chart created successfully!")
                            print(f"{'':>14}Saved as: {filename}")
                            
                            # Try to open the chart
                            if chart_path and os.path.exists(chart_path):
                                try:
                                    import subprocess
                                    import platform
                                    system = platform.system()
                                    if system == "Windows":
                                        os.startfile(chart_path)
                                    elif system == "Darwin":
                                        subprocess.run(["open", chart_path])
                                    else:
                                        subprocess.run(["xdg-open", chart_path])
                                    print(f"{'':>14}Chart opened automatically")
                                except Exception as e:
                                    print(f"{'':>14}Chart saved but couldn't auto-open: {e}")
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            print(f"{'Agent':<12}: Chart creation failed: {error_msg}")
                            if "timed out" in error_msg.lower():
                                print(f"{'':>14}Tip: The server might be busy. Try again in a moment.")
                    else:
                        print(f"{'Agent':<12}: Please provide matching labels and values")
                        print(f"{'':>14}Example: create pie chart with labels [A,B,C] values [10,20,30]")
                
                elif action == "query":
                    question = command.get("question", "")
                    if not question:
                        continue
                    
                    if not self.loaded_documents:
                        print(f"{'Agent':<12}: No documents loaded yet. Please load a PDF first!")
                        continue
                    
                    print(f"{'Agent':<12}: Analyzing your question...")
                    
                    result = await self.query_documents(question)
                    
                    if result.get("success"):
                        if result.get("found_results"):
                            confidence = result.get("confidence", 0)
                            answer = result.get("llm_answer", "")
                            
                            # Display confidence indicator
                            if confidence >= 80:
                                conf_icon = "[HIGH]"
                            elif confidence >= 60:
                                conf_icon = "[MED]"
                            else:
                                conf_icon = "[LOW]"
                            
                            print(f"{'Agent':<12}: {conf_icon} Answer (Confidence: {confidence}%):")
                            print(f"{'':>14}{answer}")
                            
                            # Check for chart generation
                            if result.get("chart_generated"):
                                chart_info = result["chart_generated"]
                                chart_filename = chart_info.get('filename', 'chart')
                                print(f"{'':>14}Generated {chart_info.get('type', 'chart')}: {chart_info.get('title', 'Chart')}")
                                print(f"{'':>14}   Saved as: {chart_filename}")
                                
                                # Try to open the chart
                                chart_path = chart_info.get('path')
                                if chart_path and os.path.exists(chart_path):
                                    try:
                                        import subprocess
                                        import platform
                                        system = platform.system()
                                        if system == "Windows":
                                            os.startfile(chart_path)
                                        elif system == "Darwin":
                                            subprocess.run(["open", chart_path])
                                        else:
                                            subprocess.run(["xdg-open", chart_path])
                                        print(f"{'':>14}   Chart opened automatically")
                                    except Exception as e:
                                        print(f"{'':>14}   Chart saved but couldn't auto-open: {e}")
                        
                        else:
                            print(f"{'Agent':<12}: I couldn't find relevant information about that topic.")
                            print(f"{'':>14}Try rephrasing your question or asking about different aspects.")
                    
                    else:
                        error_msg = result.get("error", "Unknown error occurred")
                        print(f"{'Agent':<12}: Error: {error_msg}")
                        
                        # Provide helpful tips based on error type
                        if "timed out" in error_msg.lower():
                            print(f"{'':>14}Tip: The query might be complex. Try a simpler question or wait a moment.")
                        elif "server process" in error_msg.lower():
                            print(f"{'':>14}Tip: The server may have crashed. You might need to restart the agent.")
                        elif "no response" in error_msg.lower():
                            print(f"{'':>14}Tip: Server communication issue. Try your question again.")
                
                else:
                    print(f"{'Agent':<12}: I didn't understand that. Type 'help' for available commands.")
                
            except KeyboardInterrupt:
                print(f"\n{'Agent':<12}: Goodbye!")
                self.display_session_summary()
                break
            except Exception as e:
                print(f"{'Agent':<12}: An error occurred: {e}")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.server_process:
            try:
                self.server_process.terminate()
                await asyncio.wait_for(
                    asyncio.to_thread(self.server_process.wait),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.server_process.kill()
                await asyncio.to_thread(self.server_process.wait)

async def signal_handler(agent):
    """Handle shutdown signals"""
    print("\n\nShutting down gracefully...")
    await agent.cleanup()
    sys.exit(0)

async def main():
    """Main function"""
    print("Initializing PDF RAG Conversational Agent...")
    
    agent = RAGAgent()
    
    # Set up signal handlers
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, lambda s, f: asyncio.create_task(signal_handler(agent)))
    
    try:
        # Start the MCP server
        if not await agent.start_rag_server():
            print("Failed to start RAG server. Exiting.")
            return
        
        # Initialize MCP connection
        if not await agent.initialize_mcp_connection():
            print("Failed to initialize MCP connection. Exiting.")
            return
        
        # Start interactive chat
        await agent.interactive_chat()
        
    except Exception as e:
        print(f"Agent failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    print("PDF RAG Conversational Agent")
    print("=" * 50)
    print("An intelligent assistant for PDF document analysis")
    print("Powered by Mistral LLM + Advanced RAG + Auto Charts")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Failed to start agent: {e}")