file=open("../../original_data/CAS/cas_4337.sdf")

out1 = open("out.mol","w")
out2 = open("out.label","w")

flag=true
name=""
while line=file.gets
  if flag
    name = line.chomp.strip
    flag=false
  elsif /\$\$\$\$/ =~ line
    out1.puts line
    line=file.gets
    name = line.strip
    out1.puts line
    next
    flag=true
  elsif /Ames test categorisation/ =~ line
    line=file.gets
    if /nonmutagen/ =~ line
      out2.puts "#{name} -1"
    elsif /mutagen/ =~ line
      out2.puts "#{name} 1"
    else
      STDERR.puts "error: #{line.chomp}"
    end
    next
  end
  out1.puts line
end
out1.close
out2.close

system("ruby ../convert_gspan.rb out.mol out.label")
