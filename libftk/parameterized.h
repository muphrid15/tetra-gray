#ifndef PARAMTERIZED_HDR
#define PARAMTERIZED_HDR

namespace ftk
{
	template<class Pt>
		struct Parameterizable
		{
			/*
			 template<typename B>
			 using Typecons = typename Pt::template Typecons<B>;
			 using BaseType = typename Pt::BaseType;

			 Types should define Typecons and BaseType to be used with functor.h and such.
			 This helps with detecting the proper type constructor from given data.
			 */
		};
}

#endif
