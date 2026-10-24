// One-time experiment: serialize native CT2 INT8 weights without payload copies.
#include <ctranslate2/models/model.h>
#include <ctranslate2/utils.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sys/resource.h>

namespace fs = std::filesystem;
using namespace ctranslate2;

template<typename T> T read(std::istream& in) {
  T value; in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) throw std::runtime_error("Invalid source checkpoint");
  return value;
}
std::string read_string(std::istream& in) {
  auto size=read<uint16_t>(in); std::string value(size, '\0');
  in.read(value.data(), size); if (!in || !size || value.back())
    throw std::runtime_error("Invalid checkpoint string");
  value.pop_back(); return value;
}
template<typename T> void write(std::ostream& out, T value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
void write_string(std::ostream& out, const std::string& value) {
  write<uint16_t>(out, value.size()+1); out.write(value.c_str(), value.size()+1);
}
int main(int argc, char** argv) {
  if (argc!=3) throw std::runtime_error("Usage: export_int8 SOURCE OUTPUT");
  const auto start=std::chrono::steady_clock::now();
  const fs::path source=argv[1], target=argv[2];
  if (fs::exists(target)) throw std::runtime_error("Output already exists");
  std::ifstream in(source/"model.bin", std::ios::binary);
  auto version=read<uint32_t>(in); if (version!=6) throw std::runtime_error("Requires binary v6");
  auto spec=read_string(in); auto revision=read<uint32_t>(in);
  auto count=read<uint32_t>(in); std::vector<std::string> names;
  for (uint32_t i=0; i<count; ++i) {
    names.push_back(read_string(in)); auto rank=read<uint8_t>(in);
    for (uint8_t j=0; j<rank; ++j) read<uint32_t>(in);
    read<uint8_t>(in); auto size=read<uint32_t>(in); in.seekg(size,std::ios::cur);
  }
  auto alias_count=read<uint32_t>(in);
  std::vector<std::pair<std::string,std::string>> aliases;
  for (uint32_t i=0; i<alias_count; ++i) {
    auto alias=read_string(in); auto name=read_string(in); aliases.emplace_back(alias,name);
  }
  set_num_threads(8);
  auto model=models::Model::load(source.string(),Device::CPU,0,ComputeType::INT8);
  std::vector<std::pair<std::string,const StorageView*>> values;
  std::set<std::string> seen;
  auto add=[&](const std::string& name) {
    const auto* value=model->get_variable_if_exists(name);
    if (value && seen.insert(name).second) values.emplace_back(name,value);
  };
  for (const auto& name:names) {
    add(name); add(name+"_scale"); add(name+"_compensation");
  }
  fs::create_directories(target);
  std::ofstream out(target/"model.bin",std::ios::binary);
  write<uint32_t>(out,6); write_string(out,spec); write<uint32_t>(out,revision);
  write<uint32_t>(out,values.size()); uint64_t bytes=0;
  for (const auto& pair:values) {
    const auto& name=pair.first; const auto& value=*pair.second;
    const uint64_t size=value.size()*value.item_size();
    if (size>std::numeric_limits<uint32_t>::max()) throw std::runtime_error("Tensor too large");
    write_string(out,name); write<uint8_t>(out,value.rank());
    for (auto dim:value.shape()) write<uint32_t>(out,dim);
    write<uint8_t>(out,static_cast<uint8_t>(value.dtype())); write<uint32_t>(out,size);
    out.write(static_cast<const char*>(value.buffer()),size); bytes+=size;
  }
  write<uint32_t>(out,aliases.size());
  for (const auto& pair:aliases) {write_string(out,pair.first);write_string(out,pair.second);}
  out.close(); if (!out) throw std::runtime_error("Failed to write checkpoint");
  auto config=model->config; config["quantization_policy"]="standard";
  config["matrix_dtype"]="int8"; config.erase("fp8_format");
  std::ofstream(target/"config.json") << config.dump(2) << '\n';
  for (const auto& entry:fs::directory_iterator(source)) {
    if (entry.path().filename()=="model.bin" || entry.path().filename()=="config.json") continue;
    if (entry.is_regular_file()) fs::create_symlink(fs::canonical(entry.path()),target/entry.path().filename());
  }
  rusage usage; getrusage(RUSAGE_SELF,&usage);
  auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  std::cout << "{\"conversion_seconds\":" << seconds << ",\"peak_rss_gib\":"
            << usage.ru_maxrss/1048576.0 << ",\"tensor_payload_bytes\":" << bytes
            << ",\"tensors\":" << values.size() << ",\"aliases\":" << aliases.size() << "}" << std::endl;
}
