#!/usr/bin/env python3
"""
media_audio_extraction.py
Real-world example of extracting audio from video files using Ambivo Agents
ALL CONFIGURATION FROM agent_config.yaml - NO HARDCODING

Author: Hemant Gosain 'Sunny'
Company: Ambivo
Email: sgosain@ambivo.com
License: MIT
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add the ambivo_agents package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ambivo_agents.services import create_agent_service


class MediaAudioExtractionDemo:
    """Demonstrate real audio extraction from video files"""

    def __init__(self):
        print("🎥 Initializing Media Audio Extraction Demo...")

        # Load and validate configuration FIRST
        self.load_and_validate_config()

        # Create agent service
        self.agent_service = create_agent_service()

        # Create session
        self.session_id = self.agent_service.create_session()
        self.user_id = "media_demo_user"

        # Setup directories from config
        self.setup_directories()

    def load_and_validate_config(self):
        """Load configuration from agent_config.yaml and validate required sections"""
        try:
            from ambivo_agents.config.loader import load_config, get_config_section

            print("📋 Loading configuration from agent_config.yaml...")
            self.config = load_config()

            # Validate required sections exist
            required_sections = ['media_editor', 'agent_capabilities']
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required section '{section}' in agent_config.yaml")

            # Load media processing configuration
            self.media_config = get_config_section('media_editor', self.config)

            # Validate required media processing settings
            required_media_settings = ['input_dir', 'output_dir', 'docker_image', 'timeout']
            for setting in required_media_settings:
                if setting not in self.media_config:
                    raise ValueError(f"Missing required setting 'media_editor.{setting}' in agent_config.yaml")

            # Validate capabilities
            capabilities = self.config.get('agent_capabilities', {})
            if not capabilities.get('enable_media_editor', False):
                raise ValueError(
                    "media_editor capability is not enabled in agent_config.yaml. Set capabilities.media_editor: true")

            print("✅ Configuration validated successfully")

        except FileNotFoundError:
            raise FileNotFoundError(
                "agent_config.yaml not found. Please create the configuration file with required media_editor settings."
            )
        except Exception as e:
            raise RuntimeError(f"Configuration error: {e}")

    def setup_directories(self):
        """Setup input and output directories from configuration"""
        # Get directories from config - NO HARDCODING
        self.input_dir = Path(self.media_config['input_dir'])
        self.output_dir = Path(self.media_config['output_dir'])

        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        print(f"📁 Input directory (from config): {self.input_dir.absolute()}")
        print(f"📁 Output directory (from config): {self.output_dir.absolute()}")

    async def check_media_agent_availability(self):
        """Check if media processing agent is available"""
        print("\n🔍 Checking Media Agent Availability...")

        health = self.agent_service.health_check()
        available_agents = health.get('available_agent_types', {})

        if available_agents.get('media_editor', False):
            print("✅ Media Editor Agent is available")
            return True
        else:
            print("❌ Media Editor Agent is not available")
            print("Please ensure media_editor is enabled in agent_config.yaml capabilities section")
            raise RuntimeError("Media Editor Agent not available - check agent_config.yaml configuration")

    async def extract_audio_from_video(self, video_path: str, output_format: str = "mp3",
                                       audio_quality: str = "medium"):
        """Extract audio from a video file using configuration"""
        print(f"\n🎵 Extracting audio from: {video_path}")
        print(f"📋 Format: {output_format}, Quality: {audio_quality}")

        # Validate file exists
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Get timeout from config
        timeout = self.media_config.get('timeout', 300)
        print(f"⏱️  Using timeout from config: {timeout}s")

        # Prepare the message for the media agent
        message = f"""Extract audio from the video file at path: {video_path}

Please use the following settings:
- Output format: {output_format}
- Audio quality: {audio_quality}
- Save the output to the configured media output directory: {self.output_dir}
- Use timeout: {timeout} seconds

Use the extract_audio_from_video tool to process this file."""

        start_time = time.time()

        result = await self.agent_service.process_message(
            message=message,
            session_id=self.session_id,
            user_id=self.user_id,
            conversation_id="audio_extraction_demo"
        )

        processing_time = time.time() - start_time

        if result['success']:
            print(f"✅ Audio extraction completed!")
            print(f"🤖 Agent response: {result['response']}")
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")

            # Check if output files were created in configured directory
            output_files = list(self.output_dir.glob(f"extracted_audio_*.{output_format}"))
            if output_files:
                latest_output = max(output_files, key=os.path.getctime)
                file_size = latest_output.stat().st_size
                print(f"📄 Output file: {latest_output}")
                print(f"📊 File size: {file_size / 1024 / 1024:.2f} MB")
                return str(latest_output)
            else:
                print(f"⚠️  No output files found in configured directory: {self.output_dir}")
                return None
        else:
            print(f"❌ Audio extraction failed: {result['error']}")
            return None

    async def get_media_info(self, file_path: str):
        """Get information about a media file"""
        print(f"\n📊 Getting media information for: {file_path}")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Media file not found: {file_path}")

        message = f"""Get detailed information about the media file at: {file_path}

Please use the get_media_info tool to analyze this file and provide:
- Duration
- Format details
- Audio codec and bitrate
- Video resolution (if applicable)
- File size
- Any other relevant metadata

Use the configured timeout: {self.media_config.get('timeout', 300)} seconds"""

        result = await self.agent_service.process_message(
            message=message,
            session_id=self.session_id,
            user_id=self.user_id,
            conversation_id="media_info_demo"
        )

        if result['success']:
            print(f"✅ Media info retrieved!")
            print(f"📋 Details: {result['response']}")
        else:
            print(f"❌ Failed to get media info: {result['error']}")

    async def create_sample_video(self):
        """Create a sample video file for testing using configured settings"""
        print("\n🎬 Creating sample video for testing...")

        # Use configured output directory
        sample_output_path = self.output_dir / "sample_test_video.mp4"

        message = f"""Create a simple test video file for demonstration purposes.

Please execute this FFmpeg command to create a test video in the configured output directory:

Save the output to: {sample_output_path}

Use Docker image from config: {self.media_config.get('docker_image')}
Use timeout from config: {self.media_config.get('timeout')} seconds
Use memory limit from config: {self.media_config.get('memory_limit', '2g')}

This will create a 10-second test video with a test pattern and a 1000Hz sine wave audio."""

        result = await self.agent_service.process_message(
            message=message,
            session_id=self.session_id,
            user_id=self.user_id,
            conversation_id="sample_creation_demo"
        )

        if result['success']:
            if sample_output_path.exists():
                print(f"✅ Sample video created: {sample_output_path}")
                return str(sample_output_path)
            else:
                print("⚠️  Sample video command executed but file not found in configured directory")
        else:
            print(f"❌ Failed to create sample video: {result['error']}")

        return None

    async def run_comprehensive_demo(self):
        """Run the complete media audio extraction demonstration"""
        print("\n" + "=" * 80)
        print("🎥 COMPREHENSIVE MEDIA AUDIO EXTRACTION DEMO (CONFIG-DRIVEN)")
        print("=" * 80)

        try:
            # 1. Check agent availability
            await self.check_media_agent_availability()

            # 2. Show configuration being used
            print(f"\n📋 Configuration Summary:")
            print(f"  Input Directory: {self.input_dir}")
            print(f"  Output Directory: {self.output_dir}")
            print(f"  Docker Image: {self.media_config.get('docker_image')}")
            print(f"  Timeout: {self.media_config.get('timeout')}s")
            print(f"  Memory Limit: {self.media_config.get('memory_limit', '2g')}")

            # 3. Check for existing video files in configured input directory
            video_files = list(self.input_dir.glob("*.mp4")) + list(self.input_dir.glob("*.avi"))

            if not video_files:
                print(f"\n📹 No video files found in configured input directory: {self.input_dir}")
                sample_video = await self.create_sample_video()
                if sample_video:
                    # Move sample to input directory for processing
                    input_sample = self.input_dir / "sample_test_video.mp4"
                    if Path(sample_video).exists():
                        import shutil
                        shutil.copy2(sample_video, input_sample)
                        video_files = [str(input_sample)]
                        print(f"✅ Using created sample video: {input_sample}")

                if not video_files:
                    raise RuntimeError(
                        f"No video files available for processing. Please add video files to: {self.input_dir}")

            print(f"\n📁 Found {len(video_files)} video file(s) for processing:")
            for video in video_files:
                print(f"  - {video}")

            # 4. Process first video file in detail
            primary_video = video_files[0]
            print(f"\n🎯 Processing primary video: {primary_video}")

            # Get media info first
            await self.get_media_info(primary_video)

            # Extract audio in different formats
            print("\n--- Extracting Audio in Multiple Formats ---")

            # Extract as MP3 (high quality)
            mp3_file = await self.extract_audio_from_video(
                primary_video,
                output_format="mp3",
                audio_quality="high"
            )

            # Extract as WAV (medium quality)
            wav_file = await self.extract_audio_from_video(
                primary_video,
                output_format="wav",
                audio_quality="medium"
            )

            # 5. Show final results from configured output directory
            print("\n--- Final Results ---")
            output_files = list(self.output_dir.glob("*"))
            audio_files = [f for f in output_files if f.suffix in ['.mp3', '.wav', '.flac', '.aac']]

            print(f"📊 Total output files created in {self.output_dir}: {len(output_files)}")
            print(f"🎵 Audio files: {len(audio_files)}")

            for audio_file in audio_files:
                file_size = audio_file.stat().st_size / 1024 / 1024  # MB
                print(f"  - {audio_file.name} ({file_size:.2f} MB)")

            print("\n✅ Media audio extraction demo completed successfully!")
            print(f"📁 All outputs saved to configured directory: {self.output_dir}")

        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            print("💡 Check your agent_config.yaml file for proper media_editor configuration")
            raise

    async def cleanup(self):
        """Clean up demo resources"""
        print("\n🧹 Cleaning up demo resources...")

        # Delete the demo session
        if self.session_id:
            success = self.agent_service.delete_session(self.session_id)
            if success:
                print(f"✅ Deleted demo session: {self.session_id}")

        print("✅ Cleanup completed")


async def main():
    """Main function to run the media audio extraction demo"""
    import argparse

    parser = argparse.ArgumentParser(description="Media Audio Extraction Demo (Config-Driven)")
    parser.add_argument("--input-file", help="Specific video file to process")
    parser.add_argument("--output-format", default="mp3", choices=["mp3", "wav", "aac", "flac"],
                        help="Output audio format")
    parser.add_argument("--quality", default="medium", choices=["low", "medium", "high"],
                        help="Audio quality")

    args = parser.parse_args()

    try:
        demo = MediaAudioExtractionDemo()

        if args.input_file:
            # Process specific file
            print(f"🎯 Processing specific file: {args.input_file}")

            if not Path(args.input_file).exists():
                raise FileNotFoundError(f"File not found: {args.input_file}")

            # Check agent availability
            await demo.check_media_agent_availability()

            # Get media info
            await demo.get_media_info(args.input_file)

            # Extract audio
            extracted = await demo.extract_audio_from_video(
                args.input_file,
                output_format=args.output_format,
                audio_quality=args.quality
            )

            if extracted:
                print(f"✅ Audio extracted successfully: {extracted}")
            else:
                print("❌ Audio extraction failed")
        else:
            # Run full demo
            await demo.run_comprehensive_demo()

        await demo.cleanup()

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n\n❌ Configuration Error: {e}")
        print("💡 Please check your agent_config.yaml file")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


    async def check_media_agent_availability(self):
        """Check if media processing agent is available"""
        print("\n🔍 Checking Media Agent Availability...")

        health = self.agent_service.health_check()
        available_agents = health.get('available_agent_types', {})

        if available_agents.get('media_editor', False):
            print("✅ Media Editor Agent is available")
            return True
        else:
            print("❌ Media Editor Agent is not available")
            print("Please ensure media_editor is enabled in agent_config.yaml")
            return False


    async def extract_audio_from_video(self, video_path: str, output_format: str = "mp3",
                                       audio_quality: str = "medium"):
        """Extract audio from a video file"""
        print(f"\n🎵 Extracting audio from: {video_path}")
        print(f"📋 Format: {output_format}, Quality: {audio_quality}")

        # Prepare the message for the media agent
        message = f"""Extract audio from the video file at path: {video_path}

Please use the following settings:
- Output format: {output_format}
- Audio quality: {audio_quality}
- Save the output to the media_output directory

Use the extract_audio_from_video tool to process this file."""

        start_time = time.time()

        result = await self.agent_service.process_message(
            message=message,
            session_id=self.session_id,
            user_id=self.user_id,
            conversation_id="audio_extraction_demo"
        )

        processing_time = time.time() - start_time

        if result['success']:
            print(f"✅ Audio extraction completed!")
            print(f"🤖 Agent response: {result['response']}")
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")

            # Check if output files were created
            output_files = list(self.output_dir.glob(f"extracted_audio_*.{output_format}"))
            if output_files:
                latest_output = max(output_files, key=os.path.getctime)
                file_size = latest_output.stat().st_size
                print(f"📄 Output file: {latest_output}")
                print(f"📊 File size: {file_size / 1024 / 1024:.2f} MB")
                return str(latest_output)
            else:
                print("⚠️  No output files found in the output directory")
                return None
        else:
            print(f"❌ Audio extraction failed: {result['error']}")
            return None


    async def get_media_info(self, file_path: str):
        """Get information about a media file"""
        print(f"\n📊 Getting media information for: {file_path}")

        message = f"""Get detailed information about the media file at: {file_path}

Please use the get_media_info tool to analyze this file and provide:
- Duration
- Format details
- Audio codec and bitrate
- Video resolution (if applicable)
- File size
- Any other relevant metadata"""

        result = await self.agent_service.process_message(
            message=message,
            session_id=self.session_id,
            user_id=self.user_id,
            conversation_id="media_info_demo"
        )

        if result['success']:
            print(f"✅ Media info retrieved!")
            print(f"📋 Details: {result['response']}")
        else:
            print(f"❌ Failed to get media info: {result['error']}")


    async def convert_audio_format(self, input_audio: str, target_format: str = "wav",
                                   bitrate: str = "192k"):
        """Convert audio to different format"""
        print(f"\n🔄 Converting audio format...")
        print(f"📄 Input: {input_audio}")
        print(f"🎯 Target format: {target_format}, Bitrate: {bitrate}")

        message = f"""Convert the audio file at: {input_audio}

Please convert it to:
- Format: {target_format}
- Bitrate: {bitrate}
- Sample rate: 44100 Hz

Use the convert_audio_format tool for this conversion."""

        result = await self.agent_service.process_message(
            message=message,
            session_id=self.session_id,
            user_id=self.user_id,
            conversation_id="audio_conversion_demo"
        )

        if result['success']:
            print(f"✅ Audio conversion completed!")
            print(f"🤖 Agent response: {result['response']}")

            # Check for converted files
            converted_files = list(self.output_dir.glob(f"converted_audio_*.{target_format}"))
            if converted_files:
                latest_converted = max(converted_files, key=os.path.getctime)
                print(f"📄 Converted file: {latest_converted}")
                return str(latest_converted)
        else:
            print(f"❌ Audio conversion failed: {result['error']}")

        return None


    async def adjust_audio_volume(self, input_audio: str, volume_change: str = "+5dB"):
        """Adjust audio volume"""
        print(f"\n🔊 Adjusting audio volume...")
        print(f"📄 Input: {input_audio}")
        print(f"🎚️  Volume adjustment: {volume_change}")

        message = f"""Adjust the volume of the audio file at: {input_audio}

Please apply:
- Volume change: {volume_change}
- Enable audio normalization: true

Use the adjust_audio_volume tool for this processing."""

        result = await self.agent_service.process_message(
            message=message,
            session_id=self.session_id,
            user_id=self.user_id,
            conversation_id="volume_adjustment_demo"
        )

        if result['success']:
            print(f"✅ Volume adjustment completed!")
            print(f"🤖 Agent response: {result['response']}")
        else:
            print(f"❌ Volume adjustment failed: {result['error']}")


    async def create_sample_video(self):
        """Create a sample video file for testing (using FFmpeg if available)"""
        print("\n🎬 Creating sample video for testing...")

        # Create a simple test video using FFmpeg commands
        message = """Create a simple test video file for demonstration purposes.

Please execute this FFmpeg command to create a test video:

```bash
ffmpeg -f lavfi -i testsrc=duration=10:size=320x240:rate=1 -f lavfi -i sine=frequency=1000:duration=10 -c:v libx264 -c:a aac -shortest /workspace/output/sample_test_video.mp4
```

This will create a 10-second test video with a test pattern and a 1000Hz sine wave audio."""

        result = await self.agent_service.process_message(
            message=message,
            session_id=self.session_id,
            user_id=self.user_id,
            conversation_id="sample_creation_demo"
        )

        if result['success']:
            sample_video = self.output_dir / "sample_test_video.mp4"
            if sample_video.exists():
                print(f"✅ Sample video created: {sample_video}")
                return str(sample_video)
            else:
                print("⚠️  Sample video command executed but file not found")
        else:
            print(f"❌ Failed to create sample video: {result['error']}")

        return None


    async def demonstrate_batch_processing(self, video_files: list):
        """Demonstrate batch processing of multiple video files"""
        print(f"\n📦 Batch Processing {len(video_files)} video files...")

        extracted_files = []

        for i, video_file in enumerate(video_files, 1):
            print(f"\n--- Processing File {i}/{len(video_files)} ---")

            # Extract audio
            audio_file = await self.extract_audio_from_video(
                video_file,
                output_format="mp3",
                audio_quality="high"
            )

            if audio_file:
                extracted_files.append(audio_file)

                # Get media info
                await self.get_media_info(audio_file)

            # Add delay between processing
            await asyncio.sleep(1)

        print(f"\n📊 Batch Processing Summary:")
        print(f"✅ Successfully processed: {len(extracted_files)}/{len(video_files)} files")

        return extracted_files


    async def run_comprehensive_demo(self):
        """Run the complete media audio extraction demonstration"""
        print("\n" + "=" * 80)
        print("🎥 COMPREHENSIVE MEDIA AUDIO EXTRACTION DEMO")
        print("=" * 80)

        try:
            # 1. Check agent availability
            if not await self.check_media_agent_availability():
                print("❌ Media agent not available. Exiting demo.")
                return

            # 2. Check for existing video files or create sample
            video_files = list(self.input_dir.glob("*.mp4")) + list(self.input_dir.glob("*.avi"))

            if not video_files:
                print("\n📹 No video files found in input directory.")
                sample_video = await self.create_sample_video()
                if sample_video:
                    # Move sample to input directory for processing
                    input_sample = self.input_dir / "sample_test_video.mp4"
                    if Path(sample_video).exists():
                        import shutil
                        shutil.copy2(sample_video, input_sample)
                        video_files = [str(input_sample)]
                        print(f"✅ Using created sample video: {input_sample}")

                if not video_files:
                    print("⚠️  No video files available for processing.")
                    print("Please add video files to the input directory and run again.")
                    return

            print(f"\n📁 Found {len(video_files)} video file(s) for processing:")
            for video in video_files:
                print(f"  - {video}")

            # 3. Process first video file in detail
            primary_video = video_files[0]
            print(f"\n🎯 Processing primary video: {primary_video}")

            # Get media info first
            await self.get_media_info(primary_video)

            # Extract audio in different formats
            print("\n--- Extracting Audio in Multiple Formats ---")

            # Extract as MP3 (high quality)
            mp3_file = await self.extract_audio_from_video(
                primary_video,
                output_format="mp3",
                audio_quality="high"
            )

            # Extract as WAV (medium quality)
            wav_file = await self.extract_audio_from_video(
                primary_video,
                output_format="wav",
                audio_quality="medium"
            )

            # 4. Audio format conversion
            if mp3_file:
                print("\n--- Audio Format Conversion ---")
                converted_file = await self.convert_audio_format(
                    mp3_file,
                    target_format="flac",
                    bitrate="320k"
                )

            # 5. Volume adjustment
            if wav_file:
                print("\n--- Volume Adjustment ---")
                await self.adjust_audio_volume(wav_file, volume_change="+3dB")

            # 6. Batch processing (if multiple files)
            if len(video_files) > 1:
                print("\n--- Batch Processing ---")
                batch_results = await self.demonstrate_batch_processing(video_files[1:])

            # 7. Show final results
            print("\n--- Final Results ---")
            output_files = list(self.output_dir.glob("*"))
            audio_files = [f for f in output_files if f.suffix in ['.mp3', '.wav', '.flac', '.aac']]

            print(f"📊 Total output files created: {len(output_files)}")
            print(f"🎵 Audio files: {len(audio_files)}")

            for audio_file in audio_files:
                file_size = audio_file.stat().st_size / 1024 / 1024  # MB
                print(f"  - {audio_file.name} ({file_size:.2f} MB)")

            print("\n✅ Media audio extraction demo completed successfully!")

        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


    async def cleanup(self):
        """Clean up demo resources"""
        print("\n🧹 Cleaning up demo resources...")

        # Delete the demo session
        if self.session_id:
            success = self.agent_service.delete_session(self.session_id)
            if success:
                print(f"✅ Deleted demo session: {self.session_id}")

        print("✅ Cleanup completed")


async def main():
    """Main function to run the media audio extraction demo"""
    import argparse

    parser = argparse.ArgumentParser(description="Media Audio Extraction Demo")
    parser.add_argument("--input-file", help="Specific video file to process")
    parser.add_argument("--output-format", default="mp3", choices=["mp3", "wav", "aac", "flac"],
                        help="Output audio format")
    parser.add_argument("--quality", default="medium", choices=["low", "medium", "high"],
                        help="Audio quality")

    args = parser.parse_args()

    try:
        demo = MediaAudioExtractionDemo()

        if args.input_file:
            # Process specific file
            print(f"🎯 Processing specific file: {args.input_file}")

            if not Path(args.input_file).exists():
                print(f"❌ File not found: {args.input_file}")
                return

            # Check agent availability
            if await demo.check_media_agent_availability():
                # Get media info
                await demo.get_media_info(args.input_file)

                # Extract audio
                extracted = await demo.extract_audio_from_video(
                    args.input_file,
                    output_format=args.output_format,
                    audio_quality=args.quality
                )

                if extracted:
                    print(f"✅ Audio extracted successfully: {extracted}")
                else:
                    print("❌ Audio extraction failed")
        else:
            # Run full demo
            await demo.run_comprehensive_demo()

        await demo.cleanup()

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())