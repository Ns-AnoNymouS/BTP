def searchInsert(nums, target):
    left, right = 0, len(nums)
    
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
            
    return left

def main():
    nums = input().split()
    nums = [int(num) for num in nums]
    target = int(input())
    print(searchInsert(nums, target))

if __name__ == "__main__":
    main()
    