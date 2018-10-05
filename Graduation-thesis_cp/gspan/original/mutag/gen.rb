target = Hash.new

Dir.glob("../../original_data/Mutag/mutagenesis/188/s*.pl").each do |file|
  f = open(file)
  f.each_line do |line|
    if /^active\((.+)\)/ =~ line
      target[$1] = 1
    elsif /^:- active\((.+)\)/ =~ line
      target[$1] = -1
    else
      STDERR.puts "error: #{line.chomp}"
    end
  end
  f.close
end

f=open("../../original_data/Mutag/mutagenesis/common/atom_bond.pl")

dataset = Hash.new {|h,k| h[k] = Array.new}

atom = Hash.new
atom_num = 0

while line=f.gets
  next if line.chomp.strip == ""
  if /ato*m\(([def]\d+),[def]\d+_(\d+),([^,]+),.+\)/ =~ line
    comp = $1
    next unless target.has_key?(comp)
    node = $2.to_i
    type = $3
    if !atom.has_key?(type)
      atom[type] = atom_num
      atom_num += 1
    end
    dataset[comp] << "v #{node} #{atom[type]}"
  elsif /bond\(([def]\d+),[def]\d+_(\d+),[def]\d+_(\d+),(.+)\)/ =~ line
    comp = $1
    next unless target.has_key?(comp)
    node1 = $2.to_i-1
    node2 = $3.to_i-1
    type = $4
    dataset[comp] << "e #{node1} #{node2} #{type}"
  else
    STDERR.puts "error: #{line.chomp}"
  end
end

of=open("out.gspan","w")
dataset.keys.each do |k|
  of.puts "t # 0 #{target[k]} #{k}"
  dataset[k].each do |x|
    of.puts x
  end
  of.puts
end
of.close

of=open("out.atom","w")
arr = atom.to_a.map {|x| x.reverse }.sort
arr.each do |v,k|
  of.puts "#{v} #{k}"
end
of.close
