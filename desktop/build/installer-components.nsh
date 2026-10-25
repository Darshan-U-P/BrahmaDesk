!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "nsDialogs.nsh"

!macro customHeader
  Var BrahmaLlamaServerPath
  Var BrahmaModelPath
  Var BrahmaDownloadRuntimeLater
  Var BrahmaDownloadModelLater
  Var BrahmaLlamaPathControl
  Var BrahmaModelPathControl
  Var BrahmaRuntimeLaterControl
  Var BrahmaModelLaterControl
  Var BrahmaBrowseRuntimeControl
  Var BrahmaBrowseModelControl
  Page custom BrahmaComponentsPageCreate BrahmaComponentsPageLeave
!macroend

Function BrahmaComponentsPageCreate
  !insertmacro MUI_HEADER_TEXT "AI components" "Choose existing llama.cpp and GGUF files, or configure them later."
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}

  ${NSD_CreateLabel} 0 0 100% 24u "llama.cpp runtime"
  Pop $0
  ${NSD_CreateText} 0 26u 78% 13u ""
  Pop $BrahmaLlamaPathControl
  ${NSD_CreateButton} 80% 26u 20% 13u "Browse..."
  Pop $BrahmaBrowseRuntimeControl
  ${NSD_OnClick} $BrahmaBrowseRuntimeControl BrahmaBrowseRuntime
  ${NSD_CreateCheckbox} 0 44u 100% 12u "Download a compatible runtime later"
  Pop $BrahmaRuntimeLaterControl
  ${NSD_Check} $BrahmaRuntimeLaterControl

  ${NSD_CreateLabel} 0 68u 100% 24u "GGUF model"
  Pop $0
  ${NSD_CreateText} 0 94u 78% 13u ""
  Pop $BrahmaModelPathControl
  ${NSD_CreateButton} 80% 94u 20% 13u "Browse..."
  Pop $BrahmaBrowseModelControl
  ${NSD_OnClick} $BrahmaBrowseModelControl BrahmaBrowseModel
  ${NSD_CreateCheckbox} 0 112u 100% 12u "Download a model later in Model Setup"
  Pop $BrahmaModelLaterControl
  ${NSD_Check} $BrahmaModelLaterControl

  ${NSD_CreateLabel} 0 140u 100% 32u "You can leave either component unconfigured. Brahma Desk will let you configure missing components after installation."
  Pop $0
  nsDialogs::Show
FunctionEnd

Function BrahmaBrowseRuntime
  nsDialogs::SelectFileDialog open "" "llama-server.exe|llama-server.exe|Executable files|*.exe"
  Pop $0
  ${If} $0 != ""
    ${NSD_SetText} $BrahmaLlamaPathControl $0
    ${NSD_Uncheck} $BrahmaRuntimeLaterControl
  ${EndIf}
FunctionEnd

Function BrahmaBrowseModel
  nsDialogs::SelectFileDialog open "" "GGUF models|*.gguf|All files|*.*"
  Pop $0
  ${If} $0 != ""
    ${NSD_SetText} $BrahmaModelPathControl $0
    ${NSD_Uncheck} $BrahmaModelLaterControl
  ${EndIf}
FunctionEnd

Function BrahmaComponentsPageLeave
  ${NSD_GetText} $BrahmaLlamaPathControl $BrahmaLlamaServerPath
  ${NSD_GetText} $BrahmaModelPathControl $BrahmaModelPath
  ${NSD_GetState} $BrahmaRuntimeLaterControl $BrahmaDownloadRuntimeLater
  ${NSD_GetState} $BrahmaModelLaterControl $BrahmaDownloadModelLater

  ${If} $BrahmaLlamaServerPath != ""
    IfFileExists "$BrahmaLlamaServerPath" BrahmaRuntimePathValid
    MessageBox MB_ICONEXCLAMATION "The selected llama-server.exe path does not exist."
    Abort
    BrahmaRuntimePathValid:
  ${EndIf}
  ${If} $BrahmaModelPath != ""
    IfFileExists "$BrahmaModelPath" BrahmaModelPathValid
    MessageBox MB_ICONEXCLAMATION "The selected GGUF model path does not exist."
    Abort
    BrahmaModelPathValid:
  ${EndIf}

  CreateDirectory "$APPDATA\brahma-desk"
  ClearErrors
  FileOpen $0 "$APPDATA\brahma-desk\install-selection.ini" w
  IfErrors 0 +3
    MessageBox MB_ICONSTOP "Could not save component selections in your AppData folder."
    Abort
  FileWrite $0 "[components$\r$\n"
  FileWrite $0 "llama_server_path=$BrahmaLlamaServerPath$\r$\n"
  FileWrite $0 "model_path=$BrahmaModelPath$\r$\n"
  FileWrite $0 "download_runtime_later=$BrahmaDownloadRuntimeLater$\r$\n"
  FileWrite $0 "download_model_later=$BrahmaDownloadModelLater$\r$\n"
  FileWrite $0 "]$\r$\n"
  FileClose $0
FunctionEnd
