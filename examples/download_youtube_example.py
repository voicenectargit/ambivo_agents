import asyncio
from ambivo_agents.agents.youtube_download import YouTubeDownloadAgent
from ambivo_agents.core.memory import create_redis_memory_manager


async def direct_agent_example():
    # Create agent directly
    memory = create_redis_memory_manager("youtube_agent")
    agent = YouTubeDownloadAgent("yt_001", memory)

    # Download audio
    result = await agent._download_youtube_audio("https://youtube.com/watch?v=dQw4w9WgXcQ")
    print("Audio download result:", result)

    # Download video
    result = await agent._download_youtube_video("https://youtube.com/watch?v=dQw4w9WgXcQ")
    print("Video download result:", result)

    # Get video info
    result = await agent._get_youtube_info("https://youtube.com/watch?v=dQw4w9WgXcQ")
    print("Video info:", result)


asyncio.run(direct_agent_example())