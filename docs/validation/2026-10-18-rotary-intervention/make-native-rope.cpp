#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
int main(int argc,char**argv){
 if(argc!=2)return 2;
 constexpr int positions=524288,dim=64;
 for(int trig=0;trig<2;++trig){
  std::ofstream out(std::string(argv[1])+(trig?"/sin.bin":"/cos.bin"),std::ios::binary);
  if(!out)return 3;
  for(int pos=0;pos<positions;++pos){
   uint16_t values[dim];
   for(int i=0;i<dim/2;++i){
    float angle=float(pos)/std::pow(6400000.f,float(2*i)/64);
    float value=trig?std::sin(angle):std::cos(angle);
    uint32_t bits;std::memcpy(&bits,&value,4);bits+=0x7fff+((bits>>16)&1);
    values[i]=values[i+dim/2]=uint16_t(bits>>16);
   }
   out.write(reinterpret_cast<const char*>(values),sizeof(values));
  }
  if(!out)return 4;
 }
}
