# Chapter 3: Development Environment Setup

## 3.1 Overview

This chapter guides you through setting up a complete STM32 development environment. We'll cover multiple IDE options and essential tools.

## 3.2 STM32CubeMX Installation

STM32CubeMX is a graphical configuration tool that generates initialization code for STM32 projects.

### Download and Install

1. **Visit ST Website**
   - Go to: https://www.st.com/en/development-tools/stm32cubemx.html
   - Create a free account if needed
   - Download the installer for your OS (Windows/Linux/macOS)

2. **Installation Steps**
   - Run the installer
   - Accept license agreement
   - Choose installation directory
   - Install Java Runtime Environment if prompted
   - Complete installation

3. **First Launch**
   - Start STM32CubeMX
   - Check for updates (Help → Check for Updates)
   - Download firmware packages for your MCU series

### Firmware Package Installation

STM32CubeMX uses firmware packages (HAL/LL libraries) for code generation.

**Install Packages:**
- Help → Manage Embedded Software Packages
- Select your MCU series (e.g., STM32F1, STM32F4)
- Click "Install Now"
- Wait for download to complete

**Common Packages:**
- STM32F1: For STM32F103 (Blue Pill)
- STM32F4: For STM32F407 (Discovery)
- STM32L4: For low-power applications

## 3.3 IDE Options

### Option 1: STM32CubeIDE (Recommended for Beginners)

**Advantages:**
- Free and official from ST
- Integrated with CubeMX
- Eclipse-based with good debugging
- Cross-platform (Windows/Linux/macOS)

**Installation:**

1. Download from: https://www.st.com/en/development-tools/stm32cubeide.html
2. Run installer
3. Select installation directory
4. Install ST-Link drivers when prompted
5. Launch STM32CubeIDE

**First Project:**
```
File → New → STM32 Project
→ Select your MCU or board
→ Name your project
→ Initialize peripherals with CubeMX
```

### Option 2: Keil MDK-ARM

**Advantages:**
- Industry standard
- Excellent debugging tools
- Optimized compiler
- Professional support

**Limitations:**
- Free version: 32 KB code size limit
- Windows only
- Commercial license required for full features

**Installation:**

1. Download from: https://www.keil.com/download/product/
2. Install MDK-ARM
3. Install STM32 device packs:
   - Pack Installer → STM32 → Install
4. Register for free license (32 KB limit)

**Creating Project:**
```
Project → New µVision Project
→ Select Device (STM32F103C8)
→ Manage Run-Time Environment
→ Select CMSIS and Device Startup
```

### Option 3: Visual Studio Code + PlatformIO

**Advantages:**
- Free and open source
- Modern interface
- Extensive extensions
- Cross-platform

**Installation:**

1. Install Visual Studio Code
2. Install PlatformIO extension
3. Install STM32 platform:
   ```
   PlatformIO Home → Platforms → ST STM32 → Install
   ```

**Creating Project:**
```
PlatformIO Home → New Project
→ Name: MyProject
→ Board: Select your STM32 board
→ Framework: STM32Cube or Arduino
```

## 3.4 ST-Link Driver Installation

### Windows

**Automatic Installation:**
- Usually installed with STM32CubeIDE
- Or download from: https://www.st.com/en/development-tools/stsw-link009.html

**Manual Installation:**
1. Connect ST-Link to USB
2. Device Manager → Update Driver
3. Browse to ST-Link driver folder
4. Install

**Verify Installation:**
- Device Manager → Universal Serial Bus devices
- Should see "STMicroelectronics STLink dongle"

### Linux

**Install udev rules:**
```bash
# Download udev rules
wget https://raw.githubusercontent.com/stlink-org/stlink/develop/config/udev/rules.d/49-stlinkv2.rules

# Copy to udev rules directory
sudo cp 49-stlinkv2.rules /etc/udev/rules.d/

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add user to dialout group
sudo usermod -a -G dialout $USER
```

**Install stlink tools:**
```bash
# Ubuntu/Debian
sudo apt-get install stlink-tools

# Verify
st-info --version
```

### macOS

**Install via Homebrew:**
```bash
brew install stlink
```

## 3.5 Serial Terminal Software

### Windows Options

**1. PuTTY**
- Free and lightweight
- Download: https://www.putty.org/
- Configuration: Serial, select COM port, 115200 baud

**2. Tera Term**
- Free with macro support
- Download: https://ttssh2.osdn.jp/
- Good for automation

**3. RealTerm**
- Advanced features
- Hex display
- Good for binary protocols

### Cross-Platform Options

**1. CoolTerm**
- Simple and reliable
- Download: https://freeware.the-meiers.org/

**2. Serial Monitor (Arduino IDE)**
- If Arduino IDE installed
- Tools → Serial Monitor

**3. minicom (Linux/macOS)**
```bash
# Install
sudo apt-get install minicom  # Linux
brew install minicom          # macOS

# Configure
minicom -s
→ Serial port setup
→ Set device to /dev/ttyUSB0 (Linux) or /dev/cu.usbserial (macOS)
→ Set baud rate to 115200
```

## 3.6 Additional Tools

### Logic Analyzer Software

**PulseView (Sigrok)**
- Free and open source
- Supports many protocols (I2C, SPI, UART)
- Download: https://sigrok.org/wiki/PulseView

**Saleae Logic**
- Professional tool
- Free for 8 channels
- Excellent protocol analyzers

### Version Control

**Git**
- Essential for code management
- Download: https://git-scm.com/

**Basic Setup:**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Documentation Tools

**STM32 Resources:**
- Reference Manual: Detailed peripheral descriptions
- Datasheet: Electrical characteristics, pinout
- Programming Manual: Cortex-M core details
- Application Notes: Specific topics and examples

**Download from:**
https://www.st.com/ → Select your MCU → Documentation

## 3.7 Workspace Setup

### Recommended Directory Structure

```
STM32_Projects/
├── Libraries/
│   ├── CMSIS/
│   ├── HAL_Drivers/
│   └── Third_Party/
├── Projects/
│   ├── Project1/
│   │   ├── Core/
│   │   ├── Drivers/
│   │   └── Makefile
│   └── Project2/
└── Tools/
    ├── Scripts/
    └── Utilities/
```

### Project Template

Create a template project with:
- Basic initialization code
- Common peripheral configurations
- Makefile or build scripts
- README template

## 3.8 Verification and Testing

### Test ST-Link Connection

**STM32CubeIDE:**
1. Connect ST-Link to PC and STM32 board
2. Run → Debug Configurations
3. STM32 C/C++ Application → New
4. Click "Debug" - should connect successfully

**Command Line (Linux/macOS):**
```bash
# Check ST-Link detection
st-info --probe

# Expected output:
# Found 1 stlink programmers
# version:    V2J37S27
# serial:     066DFF575251717867013127
```

### Test Serial Communication

1. Connect USB-to-Serial adapter
2. Open serial terminal
3. Configure: 115200 baud, 8N1
4. Connect to correct COM port
5. Should be ready to receive data

### Create Hello World Project

**Quick Test:**
1. Create new project in STM32CubeIDE
2. Select your MCU
3. Generate code
4. Add simple LED blink code
5. Build and flash
6. Verify LED blinks

## 3.9 Common Issues and Solutions

### ST-Link Not Detected

**Solutions:**
- Check USB cable (use data cable, not charge-only)
- Reinstall ST-Link drivers
- Try different USB port
- Check Device Manager (Windows) for errors
- Verify ST-Link firmware is up to date

### Build Errors

**Solutions:**
- Check toolchain installation
- Verify include paths
- Clean and rebuild project
- Check for missing libraries

### Upload Failures

**Solutions:**
- Verify BOOT0 pin configuration (should be LOW)
- Check SWD connections
- Ensure target has power
- Try mass erase before programming
- Check for read protection

### Serial Port Not Found

**Solutions:**
- Install USB-to-Serial drivers (CH340, CP2102, FTDI)
- Check Device Manager for COM port number
- Verify baud rate settings
- Try different USB port

## 3.10 Best Practices

### Development Workflow

1. **Plan First**
   - Define requirements
   - Choose peripherals
   - Design architecture

2. **Use Version Control**
   - Commit frequently
   - Write meaningful commit messages
   - Use branches for features

3. **Document Code**
   - Add comments for complex logic
   - Maintain README files
   - Document pin assignments

4. **Test Incrementally**
   - Test each peripheral separately
   - Use debug prints
   - Verify with oscilloscope/logic analyzer

5. **Backup Regularly**
   - Use cloud storage
   - Keep multiple backups
   - Version control is essential

## Next Steps

Proceed to [Chapter 4: Basic Programming](04-Basic-Programming.md) to start writing your first STM32 programs.
