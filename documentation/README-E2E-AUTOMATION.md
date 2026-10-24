# Brahma Desk — E2E Automation

## 1. Overview

This document describes the end-to-end (E2E) automation test kit for Brahma Desk.

Brahma Desk is a local-first Windows AI desktop application built with Electron, React, TypeScript, and a FastAPI backend. E2E testing helps validate important user workflows across the desktop interface and its supporting processes.

## 2. Test kit locations

The repository contains these test-related files:

- `BrahmaDesk-Electron-UI-Tests/RUN-UI-TESTS.ps1` — PowerShell entry point for the Electron UI test suite.
- `BrahmaDesk-Electron-UI-Tests/e2e/r
un-electron-ui-tests.cjs` — Electron E2E test runner.
- `Test-BrahmaDesk-E2E.ps1` — root-level E2E test script.
- `Test-BrahmaDesk-Backend.ps1` — root-level backend test script.

## 3. Running the tests

Run commands from the project root unless a script specifies otherwise.

### Electron UI test kit

```powershell
.\BrahmaDesk-Electron-UI-Tests\RUN-UI-TESTS.ps1
```

### Root-level E2E test

```powershell
.\Test-BrahmaDesk-E2E.ps1
```

### Backend test

```powershell
.\Test-BrahmaDesk-Backend.ps1
```

If a script fails because of PowerShell execution policy, dependency installation, or a missing executable, inspect the error and resolve the prerequisite before rerunning it. Do not bypass security controls globally just to run the tests.

## 4. Recommended validation areas

Use the test suite to validate applicable workflows such as:

- Application startup and shutdown.
- Rendering and interaction of the desktop interface.
- Chat input and response handling.
- Backend availability and error handling.
- Conversation persistence and application recovery.
- Update Center visibility and user interactions.

These are recommended validation areas, not a claim that every scenario is currently automated.

## 5. Before running tests

1. Install the project dependencies required by the test scripts.
2. Check the instructions in the relevant PowerShell script.
3. Ensure the application and backend prerequisites are available.
4. Run the test suite and review the complete output.
5. Record failures and reproduce them before changing application code.

## 6. Reporting results

A useful test report should include:

- Test command executed.
- Environment and application version.
- Passed, failed, and skipped tests, where reported.
- Relevant error messages or screenshots.
- Steps to reproduce failures.
- Whether the issue was fixed and retested.

## 7. Scope and limitations

E2E automation verifies only the scenarios actually exercised by its tests. A passing test run does not independently establish that every application feature, operating system configuration, or packaged installer works correctly.

Update this document whenever the test commands, runner, dependencies, or test coverage change.
