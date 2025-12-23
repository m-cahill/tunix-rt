import asyncio
import uuid
from unittest.mock import AsyncMock

import pytest

from tunix_rt_backend.services.tunix_execution import LogManager


@pytest.mark.asyncio
async def test_log_manager_buffering():
    mock_db = AsyncMock()
    run_id = uuid.uuid4()
    manager = LogManager(mock_db, run_id)

    # Start writer loop
    writer_task = asyncio.create_task(manager.writer_loop())

    # Feed data directly to queue to simulate stream reading
    await manager.queue.put(("stdout", "line1\n"))
    await manager.queue.put(("stdout", "line2\n"))
    await manager.queue.put(("stderr", "error1\n"))

    # Wait for processing (writer loop has timeout 0.5s or buffer 50)
    # We can force processing by waiting slightly or filling buffer?
    # Or just wait for 0.6s
    # Actually, we can cancel it and it should flush remaining items
    await asyncio.sleep(0.1)

    writer_task.cancel()
    try:
        await writer_task
    except asyncio.CancelledError:
        pass

    # Verify DB calls
    # add_all might be called multiple times or once depending on timing
    assert mock_db.add_all.called

    chunks = []
    for call in mock_db.add_all.call_args_list:
        chunks.extend(call[0][0])

    # Sort by seq to be sure
    chunks.sort(key=lambda x: x.seq)

    assert len(chunks) == 3
    assert chunks[0].chunk == "line1\n"
    assert chunks[0].seq == 1
    assert chunks[0].stream == "stdout"
    assert chunks[0].run_id == run_id

    assert chunks[1].chunk == "line2\n"
    assert chunks[1].seq == 2
    assert chunks[1].stream == "stdout"

    assert chunks[2].chunk == "error1\n"
    assert chunks[2].seq == 3
    assert chunks[2].stream == "stderr"

    # Check summary
    stdout, stderr = manager.get_summary()
    assert stdout == "line1\nline2\n"
    assert stderr == "error1\n"
