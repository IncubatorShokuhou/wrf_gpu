"""Repeat unmodified full CPS calls and verbatim original advance_ppt clears.

This wraps existing standalone drivers, never edits pristine numerical bodies.
It is imported by their optional --sequence mode.
"""
from pathlib import Path
import hashlib
import re


def bind(text,scheme):
    if scheme not in ('mskf','kfcup','camzm'):raise ValueError(scheme)
    original=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_physics_addtendc.F').read_text()
    case=original.index('   CASE (KFETASCHEME, MSKFSCHEME, KFCUPSCHEME)')
    start=original.index('           IF ( NCA(I,J) .GT. 0 ) THEN',case)
    end=original.index('\n!\n        ENDDO',start)
    clear=original[start:end]
    cup_start=original.index('             IF ( NCA(I,J) .GT. 0 ) THEN',end)
    cup_end=original.index('\n           ENDDO',cup_start)
    cup=original[cup_start:cup_end]
    declaration='''  integer :: seq_step,i,j
  integer,parameter :: k_start=1,k_end=KX,PARAM_FIRST_SCALAR=2,P_QI=2,P_QS=3
'''
    anchor='  character(len=32)::arg\n' if scheme=='camzm' else '  ! --- parse case id ---'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+declaration if scheme=='camzm' else declaration+anchor)
    call={'mskf':'  CALL MSKF_CPS(','kfcup':'  CALL KF_cup_CPS(','camzm':'  call camzm_driver( &'}[scheme]
    # Internal loop begins immediately before the real call and surrounds its
    # dumps. Initialization and input construction run only once.
    assert text.count(call)==1
    setup='''  CALL dump_col('SEQ_INPUT_W0AVG',W0AVG)
  CALL dump_col('SEQ_INPUT_W',W)
  WRITE(*,'(A,ES23.15)')'SEQ_TOP_W=',W(3,KX+1,3)
  I=3;J=3
  DO seq_step=1,10
    KTAU=seq_step
    ! Explicit caller-control cases exercise countdown expiry and a fresh call.
    IF(seq_step==2)NCA=2.*DT
    IF(seq_step==5)NCA=0.
    WRITE(*,'(A,I0)')'SEQUENCE_STEP=',seq_step
'''
    if scheme=='camzm':
        setup='''  cudt=real(stepcu)*dt/60.
  DO seq_step=1,10
    itimestep=seq_step
    WRITE(*,'(A,I0)')'SEQUENCE_STEP=',seq_step
'''
    text=text.replace(call,setup+call)
    anchor='\ncontains\n' if scheme=='camzm' else '\nCONTAINS\n'
    assert text.count(anchor)==1
    text=text.replace(anchor,'\n'+(clear if scheme!='camzm' else '')+'\n'+(cup if scheme=='kfcup' else '')+'\n  END DO\n'+anchor)
    return text,dict(advance_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        countdown_block_sha256=hashlib.sha256(clear.encode()).hexdigest(),
        cup_block_sha256=hashlib.sha256(cup.encode()).hexdigest(),
        controls='10steps fixed state; NCA=2*DT before step2 and0 before step5; original advance clear after each dump')
