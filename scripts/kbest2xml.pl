#!/data/ltutils/bin/perl

use strict;
use File::Glob ':glob';
use XML::LibXML;
use XML::Twig;

binmode STDIN,  ":encoding(utf8)";
binmode STDOUT, ":encoding(utf8)";
binmode STDERR, ":encoding(utf8)";

#my $HYPOTHESES=10;

# Perl trim function to remove whitespace from the start and end of the string
sub trim($)
{
	my $string = shift;
	$string =~ s/^\s+//;
	$string =~ s/\s+$//;
	return $string;
}

sub target_text($)
{
	my $string = shift;
	
	$string =~ s/ //g;
	return $string;
# 	
# 	my $result = "";
# 	
# #	print STDERR "doing: $string\n";
# 	
# 	my @s_ts = split " ", $string;
# 	
# #	print scalar @s_ts, " tokens\n";
# 	
# 	foreach my $s_t (@s_ts)
# 	{
# #		print "  $s_t\n";
# 		my @st = split /\|/, $s_t;
# 		
# #		print STDERR "  <$st[0]><$st[1]>\n";
# 		
# 		$result .= $st[1];
# 	}
# 	$result =~ s/_//g;
# 
# 	return $result;
}

die "Usage: kbest2xml source_xml_file results output" unless scalar @ARGV == 3;

#open SRC,  or die;
#my @srcWords = <SRC>;
#chomp @srcWords;

my $XMLParser = XML::LibXML->new( ErrorContext => 2 );

my $input = $XMLParser->parse_file( $ARGV[0] );
my @inputElements = $input->getElementsByTagName("Name");

print STDERR scalar @inputElements, " Name elements in file ", $ARGV[0], "\n";

my @srcs;

foreach my $inputElement (@inputElements)
{
	push @srcs, @{$inputElement->getChildrenByTagName("SourceName")}[0]->textContent;
# 	
# 	foreach my $tgt (@tgts)
# 	{
# 		push @inputLines, @{$inputElement->getChildrenByTagName("SourceName")}[0]->textContent;
# 	}
}

#print "There are ", scalar @inputWords, " words in the source\n";
#print join( " ", @inputWords), "\n";

my $doc  = XML::LibXML::Document->new( "1.0", "UTF-8" );
my $root = XML::LibXML::Element->new( "TransliterationTaskResults" );
$doc->setDocumentElement( $root );

$root->setAttribute( "SourceLang", $input->documentElement()->getAttribute("SourceLang") );
$root->setAttribute( "TargetLang", $input->documentElement()->getAttribute("TargetLang") );
$root->setAttribute( "GroupID", "NICT (Language Translation)" );
$root->setAttribute( "RunID", "1" );
$root->setAttribute( "RunType", "Standard" );
#$root->setAttribute( "Comments", "n/a" );

print STDERR scalar @srcs, " source lines in file ", $ARGV[0], "\n";

my %results;

open KBEST, $ARGV[1] or die "Can't open $ARGV[1]";
binmode KBEST, ":encoding(utf8)";

while( <KBEST> )
{
	chomp;
		
	my ($id, $hyp, $prob) = split /\s\|\|\|\s/, $_;	
	
	push( @{$results{$id}}, $hyp );
}

foreach my $srcline (0..$#srcs)
{
	my $src  = $srcs[$srcline];
	
	die "No results for line (0-based) $srcline" if not defined $results{$srcline};
	
	my @hyps = @{$results{$srcline}};
	
	if( scalar @hyps == 0 )
	{
		push @hyps, $src; # the news evaluation script crashes if there are no hypotheses
	}
	
#	print "src line $srcline has ", scalar @hyps, " hyps\n";
	
	my $nameElement = $root->addChild( XML::LibXML::Element->new("Name") );
	$nameElement->setAttribute( "ID", $srcline );

	my $sourceElement = $nameElement->addChild( XML::LibXML::Element->new("SourceName") );
	$sourceElement->appendTextNode($src);
		
	my @inputWords = split ' ', $src;
#	die( "$id not defined" ) unless defined $inputWords[ $id-1 ];

	print STDERR "Warning: More than one word per line for <$src>\n" unless( scalar @inputWords == 1 );

	for my $hypID (0..$#hyps)
	{		
		my $targetElement = $nameElement->addChild( XML::LibXML::Element->new("TargetName") );
		$targetElement->setAttribute( "ID", $hypID+1 );
		
#		print "hyps[$hypID]: $hyps[$hypID]\n";
#		print "target_text( $hyps[$hypID] ): ", target_text( $hyps[$hypID] ),"\n";
#		print "trim( target_text( $hyps[$hypID] ) ): ", trim( target_text( $hyps[$hypID] ) ), "\n";
				
		$targetElement->appendTextNode( trim( target_text( $hyps[$hypID] ) ) );
	}
}

die unless scalar @{$root->getElementsByTagName("Name")} == scalar @srcs;
print STDOUT $doc->toFile($ARGV[2]);
