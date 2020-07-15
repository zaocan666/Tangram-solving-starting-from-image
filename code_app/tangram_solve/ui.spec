# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['ui.py', 'game_class.py', 'generalization.py', 'pic_process.py', 'ui_basic.py', 'ui_general.py', 'ui_userDefine.py'],
             pathex=['E:\\pycharmProject\\AI\\tangram_solve'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='ui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False , icon='tangram.ico')
