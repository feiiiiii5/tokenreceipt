"""Read the input consumed by the arm64 wheel's incremental decoder."""

import json

import lldb


def record_decoder_input(frame, _location, _internal_dict):
    output = frame.FindRegister("x4").GetValueAsUnsigned()
    if output == 0:
        return False

    process = frame.GetThread().GetProcess()

    def read(address, width):
        error = lldb.SBError()
        value = process.ReadUnsignedFromMemory(address, width, error)
        if error.Fail():
            raise RuntimeError(error.GetCString())
        return value

    pc = frame.GetPCAddress().GetFileAddress()
    step = frame.FindRegister("x1").GetValueAsUnsigned()
    view = frame.FindRegister("x2").GetValueAsUnsigned()
    dtype = read(view, 4)
    size = read(view + 0x28, 8)
    if pc != 0xC6D04 or dtype != 3 or size != 1:
        raise RuntimeError(f"Unexpected decoder input: pc={pc:#x}, dtype={dtype}, size={size}")
    data = read(view + 0x18, 8)
    row = {
        "pc": hex(pc), "x1_step": step, "x2_input_view": hex(view),
        "x4_output_view": hex(output), "storage_data": hex(data),
        "input_dtype": dtype, "input_size": size, "token_id": read(data, 4),
    }
    print("CT2_DECODER " + json.dumps(row), flush=True)
    return False
