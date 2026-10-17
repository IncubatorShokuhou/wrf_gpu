"""Stage original-WRF cases from the preserved byte-identical WN3 inputs."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
source=root/'proofs/s1_nestspec/prepare_two_domain.py'
text=source.read_text()
old="SOURCE = Path('<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run')"
assert text.count(old)==1
text=text.replace(old,"SOURCE = Path('<USER_HOME>/wrf_gpu2_lanes/s1-nestspec/coupled/mp40')")
text=text.replace("EXECUTABLE = Path('<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/install_gen2_dmpar/bin/wrf')",
                  "EXECUTABLE = SOURCE / 'wrf.exe'")
text=text.replace("ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/s1-nestspec/coupled')",
                  "ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/s1-mpfix/nested_references')")
text=text.replace("CPU24 nice19 MPI1 OMP1","CPU9 nice19 MPI1 OMP1")
exec(compile(text,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
