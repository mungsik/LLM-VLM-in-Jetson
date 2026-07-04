from src.data.packing import pack_token_lists, pack_token_stream


def test_packs_and_inserts_eos_dropping_remainder():
    docs = [[1, 2, 3], [4, 5]]
    # stream = 1,2,3,eos(0),4,5,eos(0) -> blocks of 3: [1,2,3],[0,4,5]; remainder [0] dropped
    out = pack_token_lists(docs, block_size=3, eos_id=0)
    assert out == [[1, 2, 3], [0, 4, 5]]


def test_empty_input_returns_empty():
    assert pack_token_lists([], block_size=4, eos_id=0) == []


def test_all_blocks_have_exact_block_size():
    docs = [list(range(1, 21))]
    out = pack_token_lists(docs, block_size=4, eos_id=0)
    assert all(len(b) == 4 for b in out)


def test_stream_equivalent_to_list_and_accepts_generator():
    docs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    # 제너레이터 입력(스트리밍 인터페이스)에서도 리스트 버전과 동일 결과
    streamed = list(pack_token_stream((d for d in docs), block_size=3, eos_id=0))
    assert streamed == pack_token_lists(docs, block_size=3, eos_id=0)


def test_block_size_larger_than_total_yields_nothing():
    assert list(pack_token_stream(iter([[1, 2]]), block_size=100, eos_id=0)) == []
