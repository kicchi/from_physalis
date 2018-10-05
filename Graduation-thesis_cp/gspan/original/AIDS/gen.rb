label=open("../../original_data/AIDS/aids_conc_may04.txt")

num  = Hash.new(0)
dict = Hash.new

label.gets
while line=label.gets
  x,y = line.chomp.split(",")
  dict[x.strip] = y.strip
  num[y.strip] += 1
end

file=open("../../original_data/AIDS/aido99.sdf")

out = Hash.new
num = Hash.new { |h,k| h[k]=Array.new }

out["CA"] = open("ca.mol","w")
out["CM"] = open("cm.mol","w")
out["CI"] = open("ci.mol","w")


n = 4457423
lc = 0
  
flag=true
name=""
str=""
while line=file.gets
  lc += 1
  if flag
    name = line.chomp.strip
    flag=false
  elsif /\$\$\$\$/ =~ line
    str += line
    type = dict[name]
    if type != nil
      out[type].puts str
      num[type] << name
    end
    flag=true
    str=""
    next
  end
  str += line
  if lc % 1000000 == 0
    puts "#{lc} lines done (#{100*lc.to_f/n}%)"
  end
end

out["CA"].close
out["CM"].close
out["CI"].close

puts

print "CA: "
puts num["CA"].length

print "CM: "
puts num["CM"].length

print "CI: "
puts num["CI"].length

puts
print "Total "
puts num["CA"].length+num["CM"].length+num["CI"].length



out1 = open("./out_ca_vs_cm.label","w")
num["CA"].each do |x|
  out1.puts "#{x} 1"
end
num["CM"].each do |x|
  out1.puts "#{x} -1"
end
out1.close
system("cat ca.mol cm.mol > out_ca_vs_cm.mol")

out2 = open("./out_cacm_vs_ci.label","w")
(num["CA"]+num["CM"]).each do |x|
  out2.puts "#{x} 1"
end
num["CI"].each do |x|
  out2.puts "#{x} -1"
end
out2.close
system("cat ca.mol cm.mol ci.mol > out_cacm_vs_ci.mol")

out3 = open("./out_ca_vs_ci.label","w")
num["CA"].each do |x|
  out3.puts "#{x} 1"
end
num["CI"].each do |x|
  out3.puts "#{x} -1"
end
out3.close
system("cat ca.mol ci.mol > out_ca_vs_ci.mol")

puts "converting ... CA-VS-CM"
system("ruby ../convert_gspan.rb out_ca_vs_cm.mol out_ca_vs_cm.label")
system("mv out.gspan out_ca_vs_cm.gspan")
system("mv out.atom out_ca_vs_cm.atom")

puts "converting ... CA/CM-VS-CI"
system("ruby ../convert_gspan.rb out_cacm_vs_ci.mol out_cacm_vs_ci.label")
system("mv out.gspan out_cacm_vs_ci.gspan")
system("mv out.atom out_cacm_vs_ci.atom")

puts "converting ... CA-VS-CI"
system("ruby ../convert_gspan.rb out_ca_vs_ci.mol out_ca_vs_ci.label")
system("mv out.gspan out_ca_vs_ci.gspan")
system("mv out.atom out_ca_vs_ci.atom")

