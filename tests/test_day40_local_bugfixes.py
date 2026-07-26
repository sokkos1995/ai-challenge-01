import pytest
from app.services.day40_local import BrokenAvgService, BrokenSliceService, BrokenAppendService

# Test cases for BrokenAvgService
def test_broken_avg_service():
    service = BrokenAvgService()
    assert service.average([1, 2, 3]) == 2.0
    assert service.average([]) is None
    with pytest.raises(ValueError):
        service.average(None)

# Test cases for BrokenSliceService
def test_broken_slice_service():
    service = BrokenSliceService()
    assert service.slice_list([1, 2, 3], 1, 2) == [2]
    assert service.slice_list([], 0, 0) == []
    with pytest.raises(IndexError):
        service.slice_list([1, 2, 3], 3, 4)

# Test cases for BrokenAppendService
def test_broken_append_service():
    service = BrokenAppendService()
    result = service.append_item([1, 2, 3], 4)
    assert result == [1, 2, 3, 4]
    assert isinstance(result, list)
    with pytest.raises(TypeError):
        service.append_item(None, 5)

if __name__ == "__main__":
    pytest.main()
