

GBK to UTF-8 Conversion Script for .cpp Files
--------------------------------------------
This PowerShell script recursively scans all `.cpp` files in the current directory and its subdirectories
and converts those encoded in GBK to UTF-8. Files already in UTF-8 are skipped.
```powershell
Get-ChildItem -Path . -Recurse -Filter "*.cpp" | ForEach-Object { 
    $bytes = [System.IO.File]::ReadAllBytes($_.FullName); 
    $utf8String = [System.Text.Encoding]::UTF8.GetString($bytes); 
    $backToBytes = [System.Text.Encoding]::UTF8.GetBytes($utf8String); 
    if ([System.Linq.Enumerable]::SequenceEqual($bytes, $backToBytes)) { 
        Write-Host "Skipping UTF-8 file: $($_.FullName)" 
    } else { 
        $gbkString = [System.Text.Encoding]::GetEncoding("GB2312").GetString($bytes); 
        [System.IO.File]::WriteAllText($_.FullName, $gbkString, [System.Text.Encoding]::UTF8); 
        Write-Host "Converted GBK file: $($_.FullName)" 
    } 
}
```